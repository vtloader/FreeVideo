"""Reading a quantized weight cache on Ascend, dequantized to BF16 at load time.

Ascend has neither FP8 tensor cores nor the ConvRot int8 product, so a quantized
cache cannot stay quantized the way CUDA keeps it: there is no W8A16 kernel to
decode E4M3 inside the matmul, and no int8 TensorOps path. What the storage still
buys here is bytes on disk, and nothing else. The matrix is materialized as BF16
while the cache is read, so both the device footprint and the host copies the
residency paths keep are BF16 either way; only the cache bundle that had to be
downloaded and stored is smaller. That is also why the placement constants in
``policy.npu_placement`` need no re-derivation.

The modules stay ordinary ``nn.Linear``. :func:`install_cached_linears` registers
one state-dict pre-hook per target that rewrites the cache's storage keys
(``weight_fp8`` or ``weight_int8`` beside ``weight_scale``) into a plain BF16
``weight``. The shared loader in ``runtime.py`` therefore never has to know that
quantized storage exists, and the residency hooks see ordinary BF16 parameters.
"""
import torch

from .mps_int8 import CONVROT_GROUP, convrot_rotate

# Bound the FP32 tile a dequantizer materializes: a whole 18,144-row matrix in
# FP32 would be hundreds of MiB of scratch on top of the BF16 result.
SCRATCH_BYTES = 16 * 2**20


def decode_fp8(weight, scale):
    """E4M3 rows and FP32 scales to BF16, in the order the W8A16 kernel folds them.

    E4M3 to FP32, multiply the scale, then round to BF16, tile by tile. This is the
    same order ``backends/mps_weights.decode_weight`` uses, so the two decoders
    agree on the result rather than merely being close to it.
    """
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError('Expected a two-dimensional E4M3 weight')
    if scale.dtype != torch.float32 or scale.numel() not in (1, weight.shape[0]):
        raise ValueError('Expected FP32 scalar or per-output-channel FP8 scales')
    scales = scale.reshape(-1, 1)
    rows = max(1, SCRATCH_BYTES // max(1, weight.shape[1] * 4))
    result = torch.empty(weight.shape, dtype=torch.bfloat16)
    for begin in range(0, weight.shape[0], rows):
        end = begin + rows
        block = weight[begin:end].float()
        block.mul_(scales if scales.numel() == 1 else scales[begin:end])
        result[begin:end].copy_(block)
        del block
    return result


def decode_int8(weight, scale, group=None):
    """int8 rows and FP32 row scales to BF16, rotated back out of the ConvRot basis.

    The ConvRot transform is symmetric and orthogonal, so the call that rotates
    rows into that basis rotates dequantized rows back out of it; ``group=None``
    means the rows were stored in the original basis. This mirrors the inverse
    rotation ``backends/mps_weights`` applies on Macs.
    """
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError('Expected a two-dimensional int8 weight')
    if scale.dtype != torch.float32 or scale.numel() != weight.shape[0]:
        raise ValueError('Expected one FP32 scale per int8 output row')
    values = weight.float() * scale.reshape(-1, 1)
    if group is not None:
        if group != CONVROT_GROUP or weight.shape[1] % CONVROT_GROUP:
            raise ValueError('ConvRot int8 rows need 256-channel groups')
        values = convrot_rotate(values, group)
    return values.to(torch.bfloat16)


def _storage_target(spec, name, kind):
    """The storage kind and its group for one cache entry, or a refusal naming why.

    ``kind`` comes from the manifest, not from the entry: ``backends/mps_int8``
    takes ``storage`` the same way and keeps only ``convrot_group`` beside the
    shapes. An int8 entry may leave the group out, and its rows then sit in the
    original basis, which is the tolerance that module applies as well.
    """
    if spec.get('input_dtype', 'bfloat16') != 'bfloat16':
        raise ValueError('The NPU backend computes in BF16 and reads BF16 caches; ' + name
                         + ' declares input dtype ' + str(spec.get('input_dtype')))
    if kind == 'fp8':
        return 'fp8', None
    group = spec.get('convrot_group')
    if group not in (None, CONVROT_GROUP):
        raise ValueError('Unknown ConvRot group for ' + name + ': ' + str(group))
    return 'int8', group


def _rewrite_hook(kind, group):
    """Rewrite this module's storage keys into a BF16 ``weight`` entry.

    ``load_state_dict`` calls the pre-hook right before ``_load_from_state_dict``
    for the module the hook is registered on, so the swapped entries are the ones
    that module then binds. The storage keys are removed rather than left in place,
    otherwise a strict load reports them as unexpected.
    """

    def hook(*args):
        # Torch passes the state dict first by default and the module first when the
        # hook is registered with ``with_module=True``; accept either shape instead
        # of pinning this to one signature.
        state = args[1:] if isinstance(args[0], torch.nn.Module) else args
        state_dict, prefix = state[0], state[1]
        storage = prefix + ('weight_int8' if kind == 'int8' else 'weight_fp8')
        scales = prefix + 'weight_scale'
        if storage not in state_dict or scales not in state_dict:
            # Absent keys are normal: load_state_dict recurses into every module, so
            # a hook sees state dicts that do not contain its own prefix at all.
            other = prefix + ('weight_fp8' if kind == 'int8' else 'weight_int8')
            if other in state_dict:
                # But the *other* storage key being there means the manifest named a
                # format the stored bytes are not, and loading on would leave the
                # initialized weights in place while strict=False hid it.
                raise ValueError('Quantized cache entry stores ' + other.rsplit('.', 1)[-1]
                                 + ' where the manifest declared ' + kind + ' storage: ' + prefix.rstrip('.'))
            return
        stored, scale = state_dict.pop(storage), state_dict[scales].float()
        del state_dict[scales]
        state_dict[prefix + 'weight'] = (decode_int8(stored, scale, group) if kind == 'int8'
                                         else decode_fp8(stored, scale))
    return hook


def install_cached_linears(model, linears, kind, rotation=None):
    """Make each quantized cache entry load into its ``nn.Linear`` target as BF16.

    Call this before the weights are loaded; it changes no module, it only arranges
    for the dequantization to happen as the storage keys arrive. ``kind`` is the
    storage the manifest declares. Entries are validated against the module they
    target here, so a cache built for a different model fails at setup rather than
    producing quietly wrong weights.
    """
    if kind not in ('int8', 'fp8'):
        raise ValueError('Unknown quantized cache storage: ' + str(kind))
    if kind == 'int8' and rotation not in (None, {'kind': 'convrot', 'group': CONVROT_GROUP}):
        raise ValueError('Unsupported NPU int8 rotation: ' + str(rotation))
    installed = 0
    for name, spec in linears.items():
        target = model.get_submodule(name)
        if not isinstance(target, torch.nn.Linear):
            raise ValueError('Quantized cache target is not a Linear: ' + name)
        shape = spec.get('weight_shape')
        if shape is not None and list(target.weight.shape) != list(shape):
            raise ValueError('Quantized cache shape differs from the official model: ' + name)
        storage, group = _storage_target(spec, name, kind)
        target.register_load_state_dict_pre_hook(_rewrite_hook(storage, group))
        installed += 1
    return installed
