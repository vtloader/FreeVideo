"""Ascend NPU device services. Existing CUDA kernels and scheduling stay in place.

Each method maps to a ``torch.npu`` call. Services without an Ascend
implementation fail explicitly instead of falling back to CUDA, MPS or CPU, so a
porting gap is reported rather than silently changing arithmetic.
"""
from .base import BackendCapabilities, DeviceBackend


def npu_module(torch_module=None):
    """Return ``(torch, torch.npu)`` after importing Torch-NPU.

    Compiled Ascend Triton modules link ``libtorch_npu.so``, which only the
    Torch-NPU loader puts on the process search path. The device binding must
    therefore exist before the first Triton import in the process; sourcing the
    CANN environment does not substitute for it.
    """
    if torch_module is None:
        import torch as torch_module
    try:
        import torch_npu
    except ImportError as error:
        raise ImportError('The NPU device backend requires torch_npu built for this CANN and Torch version') from error
    return torch_module, torch_module.npu


def backend_source_files():
    """The sources whose change can move NPU arithmetic, as ``(name, path)`` pairs.

    The residency, hardware and monitoring modules live one directory up, so they
    are not reachable through ``with_name``. ``kernel_capabilities.COMPUTE_FILES``
    does not cover them either: that tuple is the identity of the CUDA probe
    receipt, and editing an NPU file must not invalidate a CUDA one.
    """
    from pathlib import Path
    here = Path(__file__).resolve().parent
    return ([(name, here / name) for name in
             ('__init__.py', 'base.py', 'npu.py', 'npu_attention.py', 'npu_weights.py', 'npu_vbar.py',
              'npu_linears.py')]
            + [(name, here.parent / name) for name in
               ('npu_hardware.py', 'npu_monitor.py', 'torch_device.py')])


class NPUBackend(DeviceBackend):
    """Dedicated Ascend device memory behind the shared engine contracts.

    Host pinning, streamed layer residency and prepared FP8/INT8 weight readers
    are CUDA services with no Ascend implementation yet, so both residency flags
    stay False and the corresponding calls fail instead of degrading quietly.
    """
    device = 'npu'
    capabilities = BackendCapabilities(
        name='npu', memory_model='dedicated', linear_policies=('bf16',),
        attention_candidates=('npu',), pinned_host_weights=False, streamed_weights=False)

    def __init__(self, *, torch_module=None, index=0):
        if type(index) is not int or index < 0:
            raise ValueError('NPU device index must be a nonnegative integer')
        self._torch = torch_module
        self._npu = None
        self._index = index
        self._limit = None
        self._reserve = 0

    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    @property
    def npu(self):
        if self._npu is None:
            self._torch, self._npu = npu_module(self._torch)
        return self._npu

    def is_available(self):
        return self.npu.is_available()

    def synchronize(self):
        return self.npu.synchronize()

    def empty_cache(self):
        return self.npu.empty_cache()

    def memory_info(self):
        return self.npu.mem_get_info(self._index)

    def max_memory_allocated(self):
        return self.npu.max_memory_allocated()

    def max_memory_reserved(self):
        return self.npu.max_memory_reserved()

    def memory_reserved(self):
        return self.npu.memory_reserved()

    def reset_peak_memory_stats(self):
        return self.npu.reset_peak_memory_stats()

    def host_memory_stats(self):
        """The same key set as ``torch.cuda.memory.host_memory_stats``.

        The engine reads ``allocated_bytes.current`` for its pinned host cache
        report; Torch-NPU publishes that key even where no pinning is in use.
        """
        return dict(self.npu.memory.host_memory_stats())

    def configure_budget(self, budget_bytes, allocator_limit_bytes=None, *,
                         reserve_bytes=0, capacity_trial=False):
        """Bound the process NPU allocator where CUDA Linux bounds its own.

        Windows' driver budget has no Ascend counterpart, so only an explicit
        benchmark or capacity-trial ceiling is enforced; ordinary requests leave
        the planner and the live free-memory readings in charge.
        """
        from ..gpu_budget import allocator_ceiling
        from ..policy import memory_fraction
        npu = self.npu
        total = int(npu.get_device_properties(self._index).total_memory)
        planned = memory_fraction(budget_bytes, total)
        if allocator_limit_bytes is not None and (type(allocator_limit_bytes) is not int
                                                  or not 0 < allocator_limit_bytes <= total):
            raise ValueError('Benchmark allocator limit must be positive and no larger than physical device memory')
        report = dict(device_total_bytes=total, budget_bytes=budget_bytes, planned_fraction=planned,
                      enforced=False, benchmark_allocator_limit_bytes=allocator_limit_bytes,
                      benchmark_allocator_limit_enforced=allocator_limit_bytes is not None,
                      capacity_trial_limit_enforced=False, windows_allocator_limit_enforced=False,
                      scope='PyTorch NPU caching allocator only; other Ascend allocations and '
                            'whole-device usage are not capped.')
        limit = allocator_limit_bytes
        if capacity_trial:
            free, _ = npu.mem_get_info(self._index)
            reserved = npu.memory_reserved()
            decision = allocator_ceiling(budget_bytes, total, free, reserved, reserve_bytes)
            report['admission'] = decision
            limit = min(decision['allocator_limit_bytes'], limit if limit is not None else total)
            if limit <= 0:
                raise MemoryError('NPU out of memory: no device allocator capacity remains after live budgets')
            report.update(enforced=True, capacity_trial_limit_enforced=True,
                          reason='Keep the NPU allocator within the live device budget; recover with '
                                 'controlled weight residency instead of allocator oversubscription.')
            # A resident process may already hold a larger unused pool. A new
            # fraction only affects future allocation, so release that pool once.
            if reserved > limit:
                npu.empty_cache()
                report['released_cached_pool_before_request'] = True
        if limit is not None:
            npu.set_per_process_memory_fraction(limit / total)
            self._limit, self._reserve = limit, reserve_bytes
        elif self._limit is not None:
            # A later request in the same resident worker regains ordinary
            # capacity instead of keeping yesterday's trial ceiling.
            npu.set_per_process_memory_fraction(1.)
            self._limit, self._reserve = None, 0
            report['released_capacity_trial_limit'] = True
        report['effective_allocator_limit_bytes'] = limit
        return report

    def arithmetic_identity(self):
        import hashlib
        from importlib.metadata import PackageNotFoundError, version
        from pathlib import Path
        npu = self.npu

        def package(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        return dict(device_backend='npu', npu=npu.get_device_name(self._index),
                    torch=str(self.torch.__version__),
                    packages={name: package(name) for name in ('torch-npu', 'triton-ascend')},
                    # Ascend HF32 is the dedicated-device analogue of TF32.
                    hf32=getattr(getattr(npu, 'matmul', None), 'allow_hf32', None),
                    deterministic=self.torch.are_deterministic_algorithms_enabled(),
                    backend_source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for name, path in backend_source_files()})

    def attention_kernels(self, global_backend, window_backend, *,
                          query_chunk=0, window_varlen=False):
        from .npu_attention import NPUAttentionKernels
        return NPUAttentionKernels(global_backend, window_backend,
                                   query_chunk=query_chunk, window_varlen=window_varlen)

    def prepare_linears(self, model, manifest, linear_compute, fp8_gemm):
        if fp8_gemm != 'auto':
            raise ValueError('NPU BF16 execution does not select a CUDA FP8 GEMM')
        if linear_compute not in self.capabilities.linear_policies:
            raise ValueError('Unknown linear compute policy for the NPU backend')
        precision = manifest.get('precision', 'bf16')
        if precision == 'bf16':
            # A BF16 cache stores ordinary nn.Linear weights under the standard
            # keys, so the shared loader binds them unaided. CUDA does the same.
            return None
        if precision in ('fp8', 'int8'):
            from .npu_linears import install_cached_linears
            # The manifest declares the storage; each entry only adds convrot_group.
            install_cached_linears(model, manifest.get('linears') or {}, precision,
                                   rotation=manifest.get('rotation') if precision == 'int8' else None)
            return None
        raise ValueError('The NPU backend binds BF16 weights only; this manifest stores ' + str(precision))

    def install_chunked_ff(self, module, chunk, *, recompute=False):
        if type(chunk) is not int or chunk < 1 or recompute:
            raise ValueError('NPU reference feed-forward requires a positive chunk without FP8 recompute')
        import types
        torch = self.torch
        original = module.forward

        def forward(owner, hidden, *args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError('Chunked NPU feed-forward requires inference')
            output = torch.empty_like(hidden)
            for start in range(0, hidden.shape[-2], chunk):
                output[..., start:start + chunk, :] = original(hidden[..., start:start + chunk, :], *args, **kwargs)
            return output
        module.forward = types.MethodType(forward, module)
        module._freevideo_npu_ff_chunk = chunk

    def make_offloader(self, layers, **options):
        """Per-layer residency, VBAR first.

        The placement keeps 42 of 50 blocks device-resident and streams the rest, so
        a 65.7 GiB BF16 model fits a 61.3 GiB device either way; the measured
        capacity comes from ``resident_blocks``. What the VBAR adds is the
        granularity: aimdo pages weights in and out at 32 MiB, where whole-layer
        residency can only hold or stream a layer. Without aimdo the synchronous
        host-copy residency runs instead: same arithmetic, one full copy per layer
        per step, measured at 32.65 s per step against the VBAR's 32.03 s because
        only 8 blocks are offloaded. ``stats()['residency']`` reports which one ran.
        """
        from . import npu_vbar
        from .npu_weights import NPULayerResidency
        try:
            if npu_vbar.available():
                return npu_vbar.VBARLayerResidency(layers, device=self.device, **options)
        except (ImportError, OSError, RuntimeError):
            pass
        return NPULayerResidency(layers, device=self.device, **options)

    def pin_layer_weights(self, *args, **kwargs):
        raise NotImplementedError('CUDA pinned host weights do not apply to the Ascend device backend')

    def prepare_streamed_layer(self, *args, **kwargs):
        raise NotImplementedError('NPU model streaming is not integrated yet')

    def unload_streamed_layer(self, *args, **kwargs):
        raise NotImplementedError('NPU model streaming is not integrated yet')
