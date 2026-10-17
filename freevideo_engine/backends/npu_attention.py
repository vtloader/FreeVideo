"""Ascend NPU attention with the same NHD contract as the CUDA adapter.

Torch-NPU selects the Ascend kernel behind ``scaled_dot_product_attention``, which
serves both legs, so the window leg runs the per-window loop. There is no packed
window leg. One was written as a Triton kernel that gathered the plan's rows
itself, and it was correct, but at 142 frames x 510 tokens x 56 heads x 128 it
took 2408 ms against 185 ms for the loop, and the gap was arithmetic rather than
memory: replacing both K/V loads with constants still took 1082 ms, so no
addressing, tiling or pipelining change could have closed it (consecutive K/V rows
instead of the gathered ones saved 5%; ``num_stages`` saved nothing; BLOCK_N 128
did not compile). CANN's own varlen leg, ``input_layout='TND'`` with
``actual_seq_qlen``/``actual_seq_kvlen`` -- exactly what the plan's prefix sums
encode -- fails its tiling on this device for every configuration tried, down to a
two-sequence minimum, so that route is closed too. The CUDA-only decomposed leg is
rejected instead of being emulated, so a caller sees the missing kernel rather than
different math.
"""


class NPUAttentionKernels:
    def __init__(self, global_backend, window_backend, *, query_chunk=0, window_varlen=False):
        if (global_backend, window_backend) != ('npu', 'npu'):
            raise ValueError('The NPU device requires explicit npu attention for both legs')
        if window_varlen:
            raise ValueError('Ascend has no packed window leg; use bounded window batches')
        if type(query_chunk) is not int or query_chunk < 0:
            raise ValueError('Query chunk must be nonnegative')

    def batched(self, leg, q, k, v, scale):
        if leg != 'npu':
            raise ValueError('Unknown NPU attention operator: ' + str(leg))
        if any(t.device.type != 'npu' for t in (q, k, v)):
            raise ValueError('NPU attention requires native NPU Q/K/V tensors')
        if any(t.dim() != 4 for t in (q, k, v)):
            raise ValueError('NPU attention requires NHD Q/K/V tensors')
        if len({t.dtype for t in (q, k, v)}) != 1:
            raise ValueError('NPU attention requires one dtype for Q, K and V')
        # Query rows may be a subset of the key/value rows (the dense leg of the
        # window decomposition), so only the head geometry has to agree.
        if len({t.shape[-2] for t in (q, k, v)}) != 1 or len({t.shape[-1] for t in (q, k, v)}) != 1:
            raise ValueError('NPU attention requires matching head counts and head dimensions')
        import torch.nn.functional as F
        # NHD -> NHD through BHSD: the operator consumes head-major tensors and
        # both transposes are views, so no activation copy is introduced.
        result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), dropout_p=0., is_causal=False, scale=scale)
        return result.transpose(1, 2)

    def varlen(self, *args, **kwargs):
        raise NotImplementedError('Ascend has no packed window leg; use bounded window batches')

    def decomposed(self, *args, **kwargs):
        raise NotImplementedError('CUDA FA4 decomposition is not an NPU kernel')
