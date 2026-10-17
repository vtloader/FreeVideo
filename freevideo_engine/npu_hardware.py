"""Ascend device facts for planning, beside the unchanged CUDA detector.

The shared ``Hardware`` shape lets planners and receipts keep one vocabulary:
``cuda_version`` carries the CANN runtime version on this path, ``capability``
stays ``(0, 0)`` because Ascend reports no compute capability, and the device
name identifies the SoC.
"""
from __future__ import annotations
import os
import platform
import warnings
from pathlib import Path

from .hardware import Hardware, cgroup_capacity


def cann_root():
    """The installed CANN toolkit directory, or an empty path."""
    return Path(os.environ.get('ASCEND_HOME_PATH', ''))


def cann_version():
    """CANN toolkit version from the installation layout, or an empty string."""
    return cann_root().name


def driver_version():
    """Ascend driver version from the installed version.info, or an empty string."""
    for candidate in (Path('/usr/local/Ascend/driver/version.info'),
                      cann_root().parent / 'driver' / 'version.info'):
        try:
            lines = candidate.read_text(encoding='utf-8').splitlines()
        except OSError:
            continue
        for line in lines:
            key, _, value = line.partition('=')
            if key.strip() == 'Version':
                return value.strip()
    return ''


def device_capability(npu, index):
    """Torch-NPU's compatibility reading, which Ascend leaves unset by default."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            observed = npu.get_device_capability(index)
        except (RuntimeError, ValueError):
            return (0, 0)
    if isinstance(observed, tuple) and len(observed) == 2 and all(type(v) is int for v in observed):
        return observed
    return (0, 0)


def _detect_local():
    """Device facts from the interpreter that owns the NPU context."""
    from .backends.npu import npu_module
    from .system import system_memory
    torch, npu = npu_module()
    if not npu.is_available():
        raise RuntimeError('An Ascend NPU is required for NPU generation; '
                           'plan --hardware-json can inspect presets offline.')
    index = npu.current_device()
    free, total = npu.mem_get_info(index)
    ram = system_memory()
    cap, available = cgroup_capacity(include_reclaimable=True)
    return Hardware(npu.get_device_name(index), device_capability(npu, index), total, free,
                    ram['total_bytes'],
                    min(ram['available_bytes'], available) if available is not None else ram['available_bytes'],
                    platform.system(), str(torch.__version__), cann_version(), cap,
                    str(getattr(npu.get_device_properties(index), 'uuid', '')), driver_version())


def detect():
    # End the probe's NPU context before starting the model worker: the
    # lightweight parent must not hold a device context or Torch heap alive.
    import json
    import sys
    from dataclasses import replace
    from . import subprocess_output
    from .system import system_memory
    code = ('import json; from freevideo_engine.npu_hardware import _detect_local; '
            'print(json.dumps(_detect_local().to_dict()))')
    result = subprocess_output.run_python([sys.executable, '-c', code], capture_output=True)
    if result.returncode:
        error = RuntimeError('NPU hardware detection failed: ' + result.stderr[-2000:])
        error.torch_probe_failed = True
        raise error
    observed = Hardware.from_dict(json.loads(result.stdout.strip().splitlines()[-1]))
    memory = system_memory()
    limit, available = cgroup_capacity(include_reclaimable=True)
    return replace(observed, ram_total=memory['total_bytes'],
                   ram_available=min(memory['available_bytes'], available) if available is not None else memory['available_bytes'],
                   cgroup_ram_limit=limit)


def compatibility(hardware):
    """Admission for the NPU path, not a device model or performance allowlist.

    Ascend reports no compute capability, so admission rests on a claimed device
    name; the Ascend Triton and attention kernels must still be probed
    separately before readiness is claimed.
    """
    named = bool(hardware.gpu_name)
    error = None
    if not named:
        error = ('[NPU_DEVICE_UNKNOWN] NPU detection returned no device name. Torch-NPU is '
                 'importable but no Ascend device was claimed by the runtime.')
    passed = False
    if named:
        from .kernel_capabilities import npu_receipt_readiness
        probed = npu_receipt_readiness(hardware)
        passed = bool(probed and probed['ready'])
    # ``status`` names the stage admission is at, as the CUDA shape does. A doctor
    # run re-reads this after its probes have answered the question, so it must not
    # keep reporting that the kernels are still unprobed beside kernel_probes_passed.
    status = ('ready' if passed else 'probe-required') if named else 'NPU_DEVICE_UNKNOWN'
    return dict(status=status, admitted=named,
                npu_name=hardware.gpu_name, cann_version=hardware.cuda_version,
                driver_version=hardware.driver_version, architecture='ascend',
                linear_compute='bf16' if named else None,
                kernel_probes_passed=passed,
                validation=('Ascend Triton and attention kernel probes passed on this device; the matching '
                            'probe receipt is what holds that readiness.' if passed
                            else 'Ascend Triton and attention kernel probes are required before readiness.'),
                error=error)
