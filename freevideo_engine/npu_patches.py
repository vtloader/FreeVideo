"""Patches an installed wheel needs before it can run on an Ascend NPU.

A wheel is not a checkout: there is no tree to hold a fix and no clone to re-apply it
to, and reinstalling the package would silently drop one. These patches therefore run
as an installation step against the installed file, and they are written to fail loudly
when the surrounding code has moved: an anchor that no longer matches means the
workaround needs another look, not that it can be skipped.

Each patch is idempotent -- a second run finds its own result and reports it -- so
resuming an interrupted setup is safe.
"""

from pathlib import Path


class PatchMismatch(RuntimeError):
    """The installed file no longer contains what the patch rewrites."""


# comfy_kitchen dequantizes NVFP4 by casting the block scales out of float8_e4m3fn on
# whatever device holds them. Ascend cannot perform that cast at all: every fp8 -> wider
# cast on the NPU fails with aclnnInplaceCopy 561103 (copy_d2d_baseformat_opapi), which
# is what loading the NVFP4 text encoder hit. The scales are tiny, so the conversion
# takes the CPU route and the result returns to the device the multiply happens on.
_QUANTIZATION = Path('comfy_kitchen/backends/eager/quantization.py')

_NVFP4_ANCHOR = (
    '    total_scale = per_tensor_scale.to(output_type) * '
    'block_scales_unswizzled.to(output_type)\n'
)

_NVFP4_REPLACEMENT = (
    '    # Ascend cannot cast out of float8_e4m3fn on-device: an fp8 tensor copied to\n'
    '    # bf16, fp16 or fp32 on the NPU fails with aclnnInplaceCopy 561103\n'
    '    # (copy_d2d_baseformat_opapi). These scales are tiny, so the conversion takes the\n'
    '    # CPU route, which is the one that converts, and the result returns to the device\n'
    '    # the multiplication below happens on.\n'
    '    scales = block_scales_unswizzled\n'
    "    if scales.dtype == torch.float8_e4m3fn and scales.device.type != 'cpu':\n"
    "        scales = scales.to('cpu')\n"
    '    total_scale = per_tensor_scale.to(output_type) * '
    'scales.to(output_type).to(per_tensor_scale.device)\n'
)

PATCHES = [(_QUANTIZATION, _NVFP4_ANCHOR, _NVFP4_REPLACEMENT)]


def apply(python):
    """Apply every NPU patch to the packages ``python`` can import.

    Returns the names of the files that changed, so a caller can report whether this
    run did anything. Raises :class:`PatchMismatch` when an anchor is gone.
    """
    import subprocess

    located = subprocess.run(
        [str(python), '-c',
         'import comfy_kitchen, pathlib; print(pathlib.Path(comfy_kitchen.__file__).parent)'],
        capture_output=True, text=True, check=False)
    if located.returncode:
        raise PatchMismatch('comfy_kitchen is not importable by %s: %s'
                            % (python, (located.stderr or '').strip().splitlines()[-1:]))
    package = Path(located.stdout.strip())
    changed = []
    for relative, anchor, replacement in PATCHES:
        target = package / relative.relative_to('comfy_kitchen')
        if not target.is_file():
            raise PatchMismatch('expected %s inside the installed comfy_kitchen' % relative)
        text = target.read_text(encoding='utf-8')
        if replacement in text:
            continue
        if anchor not in text:
            raise PatchMismatch(
                '%s no longer contains the line this patch rewrites; the Ascend fp8 cast '
                'workaround needs revisiting for this comfy_kitchen version' % relative)
        target.write_text(text.replace(anchor, replacement), encoding='utf-8')
        changed.append(str(relative))
    return changed
