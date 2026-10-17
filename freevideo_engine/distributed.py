"""What a request split across accelerators needs from this process.

One request runs on one accelerator today: WORLD_SIZE is 1, nothing initialises a
process group, and every caller here reads (0, 1). ``worker.rank_environment`` is
what changes that -- it pins a rank's accelerator and joins the group before any
model is built -- and the helpers below are how the rest of the engine asks.
"""


def rank_world():
    """``(rank, world_size)`` for this process; ``(0, 1)`` when not distributed.

    ``torch.distributed`` is process-global and the group is joined before the model
    loads, so reading it needs no plumbing through the call chain.
    """
    try:
        import torch.distributed as torch_distributed
        if torch_distributed.is_initialized():
            return torch_distributed.get_rank(), torch_distributed.get_world_size()
    except Exception:
        pass
    return 0, 1


def is_main():
    """Whether this process owns the request's shared paths -- its output files.

    The ranks of one split request agree on everything except where their share of a
    sequence lives, so the decode and every shared write belong to one of them. The
    rest stay for the collectives and then stop.
    """
    return rank_world()[0] == 0


_RUNTIME = None


def parallel_runtime(backend=None):
    """The Ulysses runtime for this process, or ``None`` when nothing is split.

    Built once: ``init_ulysses`` creates the group's dispatch communicators, so a second
    call would create them again. A world size of 1 returns ``None``, which is what the
    sampler expects on the single-accelerator path -- it passes no runtime at all there,
    exactly as it did before any of this existed.
    """
    global _RUNTIME
    _rank, world = rank_world()
    if world <= 1:
        return None
    if _RUNTIME is None:
        # The vendored tree is only importable once paths.puts it on sys.path, which the
        # engine's entry points do anyway; doing it here too keeps this helper from
        # depending on where it is called from.
        from .paths import add_vdn
        add_vdn()
        from src.inference.utils.ulysses_runtime import init_ulysses
        from .torch_device import device_name
        import torch
        _RUNTIME = init_ulysses(backend or ('hccl' if device_name(torch) == 'npu' else 'nccl'))
    return _RUNTIME
