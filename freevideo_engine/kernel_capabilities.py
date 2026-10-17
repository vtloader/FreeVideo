"""Execution receipts for automatic backend selection, without importing CUDA."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from .hardware import installed_backends
from .paths import data_root
from .triton_compat import temp_root

PACKAGES = ('torch', 'triton', 'triton-windows', 'diffusers', 'sageattention', 'flash-attn', 'flash-attn-4')
COMPUTE_FILES = ('attention.py', 'backends/__init__.py', 'backends/base.py', 'backends/cuda.py',
                 'backends/cuda_attention.py', 'probe.py', 'fp8_gemm.py', 'fp8_ops.py', 'int8_ops.py', 'weight_only.py',
                 'kernel_capabilities.py', 'doctor.py', 'fa4_guard.py', 'triton_compat.py', 'runtime.py',
                 'head_chunk.py', 'blocks.py', 'packing.py', 'dependencies.json')


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get('Name') or '').lower().replace('_', '-')
        if name.startswith(('nvidia-', 'cuda-')):
            result[name] = distribution.version
    return result


def identity(hardware, versions=None):
    package = Path(__file__).parent
    overlay = data_root() / 'vendor/fa4-b26-valid-tile/flash_attn/cute/flash_fwd.py'
    return dict(schema_version=1,
                hardware={k: v for k, v in hardware.to_dict().items() if k in (
                    'gpu_name', 'capability', 'vram_total', 'system', 'torch_version',
                    'cuda_version', 'gpu_uuid', 'driver_version')},
                python=str(Path(sys.executable).resolve()),
                fa4_overlay_sha256=hashlib.sha256(overlay.read_bytes()).hexdigest() if overlay.is_file() else None,
                packages=versions if versions is not None else package_versions(),
                source_sha256={name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                               for name in COMPUTE_FILES})


def receipt_path():
    return data_root() / 'kernel-capabilities.json'


def view_root(environ=None):
    """The short folder this process runs the installation from; None from the installation's own."""
    from . import launch_view
    view = launch_view.current(environ) or launch_view.current()
    return str(view.root) if view is not None and view.used else None


LINEAR_PROBES = ('linear', 'linear-int8')


def needs_path_recheck(receipt, environ=None):
    """Retry failed checks once after changing the installation or temporary view."""
    return (not readiness(receipt['kernel_probes'])['ready']
            and (receipt.get('view_root') != view_root(environ)
                 or receipt.get('temp_root') != temp_root(environ)))


def readiness(rows):
    passed = {row['backend'] for row in rows if row.get('status') == 'complete'}
    attention = sorted(passed - set(LINEAR_PROBES))
    return dict(ready='linear' in passed and bool(attention), usable_attention_backends=attention,
                int8_ready='linear-int8' in passed,
                failed_optional_backends=[row['backend'] for row in rows
                                          if row.get('status') != 'complete' and row['backend'] != 'linear'])


def matching_receipt(hardware, discovered=None):
    """The saved kernel checks when they still match this GPU, driver and packages; else None."""
    discovered = set(installed_backends() if discovered is None else discovered)
    expected = identity(hardware)
    # Normalize tuples exactly as on disk.
    expected = json.loads(json.dumps(expected))
    try:
        receipt = json.loads(receipt_path().read_text(encoding='utf-8'))
        if receipt['identity'] != expected or set(receipt['installed_attention_packages']) != discovered:
            receipt = None
        elif (not isinstance(receipt.get('kernel_probes'), list)
              or not discovered | {'linear'} <= {row.get('backend') for row in receipt['kernel_probes'] if isinstance(row, dict)}
                 <= discovered | set(LINEAR_PROBES)
              or len(receipt['kernel_probes']) != len({row.get('backend') for row in receipt['kernel_probes'] if isinstance(row, dict)})
              or any(not isinstance(row, dict) or row.get('status') not in ('complete', 'error')
                     for row in receipt['kernel_probes'])):
            receipt = None
    except (OSError, ValueError, KeyError, TypeError):
        receipt = None
    return receipt


def npu_receipt_readiness(hardware):
    """The saved Ascend kernel checks when they still match this device and code, else None.

    An Ascend device reports no compute capability, so admission cannot be
    settled from the device's own claims. The probe receipt is what tells a
    caller those kernels really ran on this device and this code.
    """
    from .doctor import npu_package_versions
    versions = dict(package_versions(), **npu_package_versions())
    expected = json.loads(json.dumps(identity(hardware, versions)))
    try:
        receipt = json.loads(receipt_path().read_text(encoding='utf-8'))
        if receipt.get('identity') != expected:
            return None
        return readiness(receipt['kernel_probes'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def available_backends(hardware, *, discovered=None, probe_missing=True, device='cuda'):
    """Reuse matching checks; re-probe after a GPU, driver or package/code change.

    Read-only planning may opt out of probing. Execution callers never turn an
    unavailable requested backend into another backend or a math SDPA kernel.
    """
    if device == 'npu':
        # Ascend attention runs through Torch-NPU's operator, not the CUDA
        # probes, so the device backend owns the candidate set here.
        from .backends import get_backend
        return set(get_backend('npu').capabilities.attention_candidates)
    discovered = set(installed_backends() if discovered is None else discovered)
    receipt = matching_receipt(hardware, discovered)
    if receipt is not None and probe_missing and needs_path_recheck(receipt):
        # They failed while the installation ran from another folder, and compilers take some
        # paths only as that folder spells them. What passed is kept wherever it ran from.
        receipt = None
    if receipt is None:
        if not probe_missing:
            return discovered
        from .doctor import doctor
        receipt = doctor(probe=True, hardware=hardware, backends=discovered)
    checked = readiness(receipt['kernel_probes'])
    if not checked['ready']:
        raise RuntimeError('GPU kernel checks did not find a working linear and attention path. '
                           'Run freevideo doctor --probe; details: ' + str(receipt_path()))
    return discovered.intersection(checked['usable_attention_backends'])


def fp8_implementation(system, scale_granularity, requested='auto'):
    if requested not in ('auto', 'torch', 'scaled-mm-epilogue'):
        raise ValueError('Unknown FP8 GEMM implementation: ' + requested)
    if requested == 'scaled-mm-epilogue' and scale_granularity != 'rowwise':
        raise ValueError('The FP8 compatibility implementation is only for rowwise caches; use native Torch for tensorwise scales')
    if requested != 'auto':
        return requested
    return 'scaled-mm-epilogue' if system == 'Windows' and scale_granularity == 'rowwise' else 'torch'
