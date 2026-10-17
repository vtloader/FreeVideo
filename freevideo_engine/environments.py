"""Pinned installation layouts; model lifetimes are independent of venv count."""
import os
from pathlib import Path
import platform

ENVIRONMENTS = {
    'unified': {'cuda': 'cu130', 'torch': ['torch==2.13.0', 'torchvision==0.28.0', 'torchaudio==2.11.0']},
    'engine': {'cuda': 'cu129', 'torch': ['torch==2.13.0', 'torchvision==0.28.0']},
    'encoder': {'cuda': 'cu130', 'torch': ['torch==2.14.0', 'torchvision==0.29.0', 'torchaudio==2.11.0']},
}

# Ascend reuses the interpreter's Torch and Torch-NPU instead of downloading a CUDA
# wheel set, so this constraints file names what the venv must not acquire (the
# CUDA Triton) rather than what it installs.
NPU_RUNTIME = 'npu-runtime.txt'
CUDA_WHEEL_PACKAGES = ('torch', 'torchvision', 'torchaudio')
CUDA_TRITON_PACKAGES = ('triton', 'triton-windows')


def setup_device(environ=None):
    """The accelerator an installation targets; ``FREEVIDEO_SETUP_DEVICE`` selects it.

    The installer has no probe of its own. An Ascend venv reuses the interpreter
    that already carries the CANN-matched Torch-NPU, so the device has to be named
    before that venv exists, and naming it is what keeps every CUDA default intact
    for callers that do not.
    """
    values = os.environ if environ is None else environ
    return (values.get('FREEVIDEO_SETUP_DEVICE') or 'cuda').strip().lower()


# The CPU wheel index. The interpreter's Torch is a +cpu build, and the default PyPI
# wheels of these two are CUDA builds whose extension refuses to register against it
# ("operator torchvision::nms does not exist"), so the +cpu variants are the only
# ones an Ascend environment can load.
TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'


def reused_torch_companions():
    """The torch stack's companions, for an environment that reuses Torch itself.

    Ascend takes Torch and Torch-NPU from the interpreter, but torchvision and
    torchaudio are ordinary wheels that no interpreter provides, and the pinned H3
    text encoder imports both at module scope (comfy/ldm/cascade, comfy/audio_encoders).
    The versions are the ones this project pairs with that Torch, taken as the +cpu
    build that matches the +cpu Torch already present.
    """
    return [item + '+cpu' for item in ENVIRONMENTS['unified']['torch'] if not item.startswith('torch==')]


def fp8_linear_available(device):
    """Whether this device family runs FP8 linear work at all.

    A quantized cache exists for FP8 tensor cores, and a device without them decodes
    that storage back to BF16 before any matrix runs, so preparing one costs a
    quantization pass and a smaller file for no gain. Ascend's linear policies are
    BF16 only. A device whose capabilities cannot be read keeps the previous answer,
    so a failure never silently changes what CUDA does.
    """
    try:
        from .backends import get_backend
        policies = get_backend(device).capabilities.linear_policies
    except Exception:
        return True
    return any('fp8' in str(policy) for policy in policies)


def prepare_device(explicit=None):
    """The accelerator a ``prepare`` run quantizes on; that command names none.

    Detection follows the order the engine's backends do, and a host with neither
    accelerator keeps the CUDA answer, which is what that command did before this
    existed. Only availability is queried, so no device is initialized by asking.
    """
    if explicit:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        npu = getattr(torch, 'npu', None)
        if npu is not None and npu.is_available():
            return 'npu'
    except Exception:
        pass
    return 'cuda'


def constraints_file(name, system=None, device=None):
    if (system or platform.system()) == 'Darwin':
        if name != 'unified':
            raise ValueError('Mac requires the native unified environment')
        from .macos_bootstrap import constraints
        return constraints()
    if (device or setup_device()) == 'npu':
        if name != 'unified':
            raise ValueError('An Ascend installation keeps the single unified environment')
        return Path(__file__).resolve().parent.parent / 'constraints' / NPU_RUNTIME
    prefix = 'windows-' if (system or platform.system()) == 'Windows' else ''
    return Path(__file__).resolve().parent.parent / 'constraints' / (prefix + name + '.txt')


def runtime_requirements(device=None):
    """The ``runtime`` extra, without the CUDA Triton requirement on Ascend.

    A constraints file can only pin what is already required, so it cannot keep
    ``triton`` out of the resolution. An Ascend installation therefore asks for the
    extra's requirements directly, minus the CUDA Triton, and takes the fork from
    the vendored wheel in ``constraints/npu-runtime.txt``.
    """
    import tomllib
    manifest = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    spec = tomllib.loads(manifest.read_text(encoding='utf-8'))
    requirements = list(spec['project']['optional-dependencies']['runtime'])
    if (device or setup_device()) != 'npu':
        return requirements
    return [item for item in requirements if not item.startswith(CUDA_TRITON_PACKAGES)]


# uv reads each value of these options as a space-separated list of files, even
# when it arrives as one quoted argument: `-c "D:\Comfy UI\pins.txt"` names
# `D:\Comfy` and `UI\pins.txt` and stops with "File not found" (exit code 2).
# Requirement files, editables, wheels and --python keep their spaces.
UV_FILE_LIST_OPTIONS = frozenset(('-c', '--constraint', '--constraints', '-b', '--build-constraint',
                                  '--build-constraints', '--override', '--overrides'))


def uv_file_arguments(command, cwd=None):
    """Return (command, cwd) for which uv reads constraint files in folders with spaces.

    Such a file is named relative to its folder, which becomes uv's working
    directory. FreeVideo passes every other path to uv as an absolute path.
    """
    command = [str(item) for item in command]
    if not command or Path(command[0]).name.lower() not in ('uv', 'uv.exe') or command[1:2] != ['pip']:
        return command, cwd
    values = {}
    for index, item in enumerate(command):
        option, equals, value = item.partition('=')
        if item in UV_FILE_LIST_OPTIONS and index + 1 < len(command):
            values[index + 1] = ('', command[index + 1])
        elif equals and option.startswith('--') and option in UV_FILE_LIST_OPTIONS:
            values[index] = (option + '=', value)
    spaced = {index: row for index, row in values.items() if ' ' in row[1]}
    if not spaced:
        return command, cwd
    base = Path(cwd or os.getcwd())
    folder = (base / next(iter(spaced.values()))[1]).parent
    result = list(command)
    for index, (prefix, value) in spaced.items():
        try:
            relative = os.path.relpath(base / value, folder)
        except ValueError:  # Another Windows drive; uv reports the original path.
            return command, cwd
        if ' ' in relative:
            return command, cwd
        result[index] = prefix + relative
    return result, str(folder)


def bootstrap_versions(value, system=None):
    result = dict(value, target_system=system or platform.system())
    if result['target_system'] == 'Darwin':
        from .macos_bootstrap import versions
        return versions(value)
    if result['target_system'] == 'Windows':
        result['uv'] = result['windows']['uv']
    return result


def environment_names(layout):
    if layout == 'unified':
        return ('unified',)
    if layout == 'dual':
        return ('engine', 'encoder')
    raise ValueError('Unknown environment layout: ' + str(layout))


def select_layout(requested=None, saved=None):
    saved = saved or {}
    layout = requested or saved.get('pending_environment_layout') or saved.get('environment_layout')
    if layout is None and saved.get('python'):
        # v0.2.0/0.2.1 receipts did not record a layout. Preserve them on update.
        layout = 'unified' if saved.get('python') == saved.get('comfy_python') else 'dual'
    layout = layout or 'unified'
    environment_names(layout)
    return layout


def role_pythons(root, layout, system=None):
    from .system import venv_python
    names = environment_names(layout)
    return {'engine': venv_python(root / 'envs' / names[0], system),
            'encoder': venv_python(root / 'envs' / names[-1], system)}
