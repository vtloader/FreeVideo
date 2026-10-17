"""Native encoder registration for our immutable read-only checkpoint maps."""
from .torch_device import device_module, device_name
from contextlib import contextmanager
import ctypes
import sys

from .encoder_checkpoint import owns_readonly_tensor


READ_ONLY_SUPPORTED = 113  # cudaDevAttrHostRegisterReadOnlySupported
PORTABLE_READ_ONLY = 0x01 | 0x08


def _registration_flags():
    """Query the existing CUDA context, without allocating or changing devices."""
    try:
        driver = (ctypes.WinDLL('nvcuda.dll') if sys.platform == 'win32'
                  else ctypes.CDLL('libcuda.so.1'))
        current = driver.cuCtxGetDevice
        current.argtypes, current.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        attribute = driver.cuDeviceGetAttribute
        attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        attribute.restype = ctypes.c_int
        device = ctypes.c_int()
        if current(ctypes.byref(device)):
            return None
        supported = ctypes.c_int()
        if attribute(ctypes.byref(supported), READ_ONLY_SUPPORTED, device.value) == 0 and supported.value:
            return PORTABLE_READ_ONLY
        host_tables = ctypes.c_int()
        if attribute(ctypes.byref(host_tables), 100, device.value) == 0 and host_tables.value:
            return 0x01  # Host page tables support read-only mappings directly.
    except (AttributeError, OSError):
        pass
    return None


def _count(manager, name):
    field = 'FREEVIDEO_READONLY_PIN_' + name
    setattr(manager, field, getattr(manager, field, 0) + 1)


def _windows_pin_limit(manager):
    """Bound new registrations by free physical RAM and commit, not total RAM.

    Registered read-only maps can become resident only during the forward.
    Reserve their bytes up front even when the initial RSS is still small.
    Existing registrations stay native-owned and remain available for unpin.
    """
    from .system import system_memory
    owned = max(0, int(manager.TOTAL_PINNED_MEMORY))
    try:
        memory = system_memory()
        free = memory.get('physical_available_bytes', memory['available_bytes'])
        commit = memory.get('commit_available_bytes')
        if commit is not None:
            free = min(free, commit)
        return min(int(manager.MAX_PINNED_MEMORY), owned + max(0, free - 2 * 2**30))
    except (OSError, RuntimeError, TypeError, KeyError, ValueError):
        _count(manager, 'MEMORY_QUERY_FAILURES')
        return owned  # Registration is optional; keep pageable inference usable.


@contextmanager
def readonly_pinning(torch, manager):
    """Adapt native pinning only while the isolated encoder is running.

    CUDA requires ReadOnly for read-only CPU mappings on platforms without
    host page tables. Native writable tensors keep the original code path;
    native registries and unpinning continue to own successful registrations.
    """
    original = manager.pin_memory
    if getattr(original, '_freevideo_readonly_adapter', False) is True:
        yield
        return
    unsupported = False
    support_checked = False
    flags = None
    pin_limit = _windows_pin_limit(manager) if sys.platform == 'win32' else None
    if pin_limit is not None:
        manager.FREEVIDEO_READONLY_PIN_CAPACITY_BYTES = pin_limit

    def clear_error(runtime):
        # Failed registration can also leave an asynchronous CUDA error. Use
        # Comfy's own drain before returning to its pageable transfer path.
        drain = getattr(manager, 'discard_cuda_async_error', None)
        if callable(drain):
            drain()
        else:
            clear = getattr(runtime, 'cudaGetLastError', None)
            if callable(clear):
                clear()

    def failed(runtime, code):
        nonlocal unsupported
        _count(manager, 'FAILURES')
        manager.FREEVIDEO_READONLY_PIN_LAST_ERROR_CODE = code
        if code in (2, 712, 801):
            clear_error(runtime)
            if code == 801:
                unsupported = True
            # Capacity, an overlapping registered page, or unavailable support.
            # No copy is needed: native offload also accepts pageable weights.
            return False
        description = {700: 'illegal memory access', 702: 'launch timeout',
                       710: 'device-side assert', 719: 'unspecified launch failure',
                       999: 'unknown error'}.get(code, 'host registration failed')
        error = RuntimeError('CUDA error: %s (read-only encoder registration, code %s)' % (description, code))
        error.error_code = code
        raise error

    def pin(tensor):
        nonlocal unsupported, support_checked, flags
        if not owns_readonly_tensor(tensor):
            return original(tensor)
        if unsupported or manager.MAX_PINNED_MEMORY <= 0 or not tensor.is_contiguous() or tensor.is_pinned():
            _count(manager, 'SKIPS')
            return False
        size, pointer = tensor.nbytes, tensor.data_ptr()
        if not pointer:
            _count(manager, 'SKIPS')
            return False
        if pin_limit is not None:
            live_limit = min(pin_limit, _windows_pin_limit(manager))
            if manager.TOTAL_PINNED_MEMORY + size > live_limit:
                _count(manager, 'MEMORY_SKIPS')
                _count(manager, 'SKIPS')
                return False
        memory = getattr(getattr(manager, 'comfy', None), 'memory_management', None)
        release = getattr(memory, 'extra_ram_release', None)
        if callable(release):
            release(memory.RAM_CACHE_HEADROOM)
        registerable = getattr(manager, 'ensure_pin_registerable', None)
        if callable(registerable):
            admitted = registerable(size)
        else:
            admitted = manager.TOTAL_PINNED_MEMORY + size <= manager.MAX_PINNED_MEMORY
        if not admitted:
            _count(manager, 'SKIPS')
            return False
        if not support_checked:
            support_checked = True
            flags = _registration_flags()
            manager.FREEVIDEO_READONLY_PIN_FLAGS = flags or 0
            if flags is None:
                unsupported = True
                _count(manager, 'SKIPS')
                return False
        runtime = getattr(device_module(torch), 'cudart', None)
        if runtime is None:
            # Ascend has no cudart: host pinning there goes through the
            # comfy-aimdo host buffer instead of this CUDA-only path.
            _count(manager, 'SKIPS')
            return False
        _count(manager, 'ATTEMPTS')
        try:
            code = int(runtime.cudaHostRegister(pointer, size, flags))
        except RuntimeError as error:
            code = getattr(error, 'error_code', None)
            if code not in (2, 712, 801):
                raise
            return failed(runtime, code)
        if code:
            return failed(runtime, code)
        manager.PINNED_MEMORY[pointer] = size
        manager.TOTAL_PINNED_MEMORY += size
        _count(manager, 'SUCCESSES')
        return True

    pin._freevideo_readonly_adapter = True
    manager.pin_memory = pin
    try:
        yield
    finally:
        manager.pin_memory = original
