"""The torch accelerator module a process is running on.

torch_npu mirrors the CUDA surface these callers use (mem_get_info, memory
accounting, synchronize, reset_peak_memory_stats, get_device_properties), so the
paths that only need that surface stay device agnostic through one lookup
instead of naming torch.cuda.
"""


def device_module(torch):
    """The torch accelerator module for this process (torch.npu or torch.cuda)."""
    npu = getattr(torch, 'npu', None)
    if npu is not None and npu.is_available():
        return npu
    return torch.cuda


def device_name(torch):
    """The device string to move host tensors onto."""
    return 'npu' if device_module(torch) is not torch.cuda else 'cuda'
