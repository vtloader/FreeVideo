"""Matched backend experiments. Sampling and full-request timing stay separate."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

# detect_hardware picks the detector by installation; imported under the name the rest
# of this file already calls.
from .cli import detect_hardware as detect
from .geometry import geometry
from .kernel_capabilities import available_backends
from .policy import choose
from .monitoring import save
from .locking import runtime_lock, LOCK_ENV
from . import processes
from .system import windows


def main(arguments=None):
    parser = argparse.ArgumentParser(prog='freevideo bench', description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--conditioning', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', help='global/window pairs; default compares installed candidates')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--mode', choices=['sampling', 'full'], default='sampling')
    parser.add_argument('--require-ram-limit', action='store_true', help='Require a dedicated zero-swap cgroup in full mode')
    parser.add_argument('--vram-gib', type=float, help='Optional capacity cap; default detects current GPU resources')
    parser.add_argument('--ram-gib', type=float, help='Optional capacity cap; default detects current RAM resources')
    parser.add_argument('--seed', type=int, default=2026090901)
    parser.add_argument('--save-best-profile', type=Path, help='Export the fastest measured sampling policy for --profile')
    args = parser.parse_args(arguments)
    if args.repeats < 1 or not 1 <= args.warmup_steps <= 2:
        parser.error('Use at least one repeat and one or two warmup steps')
    if args.require_ram_limit and args.mode != 'full':
        parser.error('--require-ram-limit is for full mode; use an outer capacity harness for sampling')
    if windows() and args.require_ram_limit:
        parser.error('Native Windows reports observed RAM; Linux cgroup hard limits are unavailable.')
    if args.save_best_profile and args.mode != 'sampling':
        parser.error('--save-best-profile currently exports sampling comparisons')
    if args.save_best_profile and args.save_best_profile.exists():
        parser.error('--save-best-profile already exists')
    args.out.mkdir(parents=True, exist_ok=False)
    hardware = detect()
    from .cli import installed_device
    available = available_backends(hardware, device=installed_device())
    if args.backends is None:
        candidates = ['cudnn/fa4', 'cudnn/sage2', 'sage2/sage2', 'cudnn/cudnn', 'torch-flash/torch-flash']
        if 'fa2' in available:
            candidates += ['cudnn/fa2', 'fa2/fa2']
        args.backends = [name for name in candidates if all(leg in available for leg in name.split('/'))]
    if args.profile:
        profile = json.loads(args.profile.read_text(encoding='utf-8'))
        if 'gpu_budget_bytes' in profile:
            profile = dict(engine=profile['engine'], decoder=profile['decoder'],
                           gpu_budget_gb=profile['gpu_budget_bytes'] / 1e9,
                           inference_ram_budget_gb=profile['ram_budget_bytes'] / 1e9,
                           allocator_config=profile['allocator_config'], policy=profile)
    else:
        profile = choose(hardware, vram_gib=args.vram_gib, ram_gib=args.ram_gib,
                         available_backends=available, canvas=geometry(frames=243)).legacy_profile()
    if args.base:
        profile['engine']['base'] = str(args.base.resolve())
    if args.checkpoint:
        profile['engine']['checkpoint'] = str(args.checkpoint.resolve())
    os.environ['PYTORCH_ALLOC_CONF'] = profile['allocator_config']
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = profile['allocator_config']
    report = {'mode': args.mode, 'hardware': hardware.to_dict(), 'profile': profile,
              'requested_backends': args.backends,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(__file__).parent.glob('*.py'))},
              'versions': {},
              'seed': args.seed, 'frames': 243, 'resolution': [1344, 768], 'steps': 8,
              'conditioning_sha256': hashlib.sha256(args.conditioning.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256((args.cache / 'manifest.json').read_bytes()).hexdigest(),
              'scope': 'Identical model/cache/conditioning/seed and resource policy; only attention changes. Sampling excludes encoding/loading/decode. Full mode includes loading/decode, but uses cached conditioning.',
              'capacity_scope': 'Placement budgets plus observed memory. Hard-limit claims require a matching outer MPS/cgroup measurement.',
              'rows': [], 'warmups': [], 'success': False}
    for name in ('torch', 'triton', 'triton-windows', 'sageattention', 'flash-attn', 'flash-attn-4', 'nvidia-cudnn-cu12',
                 'nvidia-cutlass-dsl', 'cuda-bindings', 'cuda-core'):
        try:
            report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][name] = None
    save(args.out / 'comparison.json', report)
    if args.mode == 'sampling':
        sampling(args, profile, report)
    else:
        full(args, profile, report)
    if not report['success']:
        raise SystemExit(1)
    if args.save_best_profile:
        eligible = {name: seconds for name, seconds in report['sampling_medians_seconds'].items()
                    if all(row['gpu_peak_bytes'] <= profile['gpu_budget_gb'] * 1e9
                           for row in report['rows'] if row['backend'] == name)}
        if not eligible:
            raise RuntimeError('No backend stayed within the requested observed GPU budget')
        selected = copy.deepcopy(profile)
        selected['engine']['attention'] = min(eligible, key=eligible.get)
        if 'policy' in selected:
            selected['policy']['engine'] = copy.deepcopy(selected['engine'])
        selected['selection'] = {'comparison': str((args.out / 'comparison.json').resolve()),
                                 'hardware': hardware.to_dict(), 'versions': report['versions'],
                                 'scope': 'Measured sampling only; complete-video capacity/quality validation remains separate.'}
        save(args.save_best_profile, selected)


def sampling(args, profile, report):
    import torch
    from .runtime import Engine
    from .locking import runtime_lock
    from .monitoring import device_monitor
    torch.set_num_threads(8)
    options = copy.deepcopy(profile['engine'])
    options['attention'] = args.backends[0]
    monitor = device_monitor(args.out / 'gpu.csv')
    if monitor is None:
        raise RuntimeError('Comparing backends needs a whole-device sampler; '
                           'this host has neither NVML nor DCMI.')
    engine = None
    try:
        with runtime_lock():
            report['current_attempt'] = {'phase': 'loading', 'backend': args.backends[0]}
            save(args.out / 'comparison.json', report)
            engine = Engine(args.cache, **dict(options, canvas=geometry(frames=243)))
            report['load_seconds'] = engine.load_seconds
            report['load_gpu_peak_bytes'] = monitor.peak
            working = []
            for backend in args.backends:
                row = {'backend': backend, 'completed_steps': args.warmup_steps}
                report['current_attempt'] = {'phase': 'warmup', 'backend': backend}
                save(args.out / 'comparison.json', report)
                try:
                    engine.attention.select_backend(backend)
                    _warmup(engine, args.conditioning, args.seed, args.warmup_steps)
                    row['status'] = 'complete'
                    working.append(backend)
                except Exception as error:
                    row.update(status='error', error=f'{type(error).__name__}: {error}')
                    # CUDA execution failures can poison a context; don't rank
                    # later work performed in a failed context as valid evidence.
                    if not isinstance(error, (ImportError, ModuleNotFoundError, ValueError)):
                        report['warmups'].append(row)
                        save(args.out / 'comparison.json', report)
                        raise
                report['warmups'].append(row)
                save(args.out / 'comparison.json', report)
            for repeat in range(args.repeats):
                order = working if repeat % 2 == 0 else list(reversed(working))
                for backend in order:
                    report['current_attempt'] = {'phase': 'sampling', 'backend': backend, 'repeat': repeat}
                    save(args.out / 'comparison.json', report)
                    engine.attention.select_backend(backend)
                    engine.config['attention'] = backend
                    monitor.peak = monitor.sample()[0]
                    video, audio, metrics = engine.sample(args.conditioning, args.seed)
                    finite = bool(torch.isfinite(video).all() and torch.isfinite(audio).all())
                    if not finite:
                        raise RuntimeError('Non-finite generated latents for ' + backend)
                    del video, audio
                    row = dict(backend=backend, repeat=repeat, status='complete', metrics=metrics,
                               gpu_peak_bytes=monitor.peak, actual_config=dict(engine.config),
                               finite_latents=finite)
                    report['rows'].append(row)
                    save(args.out / 'comparison.json', report)
                    print(json.dumps({'event': 'sample_complete', 'backend': backend, 'repeat': repeat,
                                      'seconds': metrics['sample_seconds'], 'gpu_peak_bytes': monitor.peak}), flush=True)
            medians = {backend: statistics.median(r['metrics']['sample_seconds'] for r in report['rows'] if r['backend'] == backend)
                       for backend in working}
            report['sampling_medians_seconds'] = medians
            report['fastest_measured_backend'] = min(medians, key=medians.get) if medians else None
            report['success'] = len(working) == len(args.backends) and bool(working)
            report['current_attempt'] = None
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.close()
        report['monitor'] = monitor.stop()
        report['monitor']['gpu_peak_bytes'] = max([report.get('load_gpu_peak_bytes', 0)] + [r['gpu_peak_bytes'] for r in report['rows']] + [monitor.peak])
        save(args.out / 'comparison.json', report)


def _warmup(engine, conditioning, seed, steps):
    class Warmed(Exception):
        pass
    count = 0
    def before(module, inputs):
        nonlocal count
        count += 1
        if count > steps:
            raise Warmed()
    hook = engine.transformer.register_forward_pre_hook(before, prepend=True)
    try:
        try:
            engine.sample(conditioning, seed)
        except Warmed:
            pass
        else:
            raise RuntimeError('Warmup did not stop at the requested step boundary')
    finally:
        hook.remove()


def full(args, profile, report):
    with runtime_lock() as descriptor:
        return full_locked(args, profile, report, descriptor)


def full_locked(args, profile, report, descriptor):
    from .paths import base_path
    base = args.base or Path(profile['engine'].get('base', base_path()))
    report['cache_policy'] = ('Windows: natural OS file cache; no eviction or cold-cache claim. Fresh compiler directories per request.' if windows() else
        'Model file pages evicted and checked before every request; separate fresh compiler directories for each request.')
    try:
        for repeat in range(args.repeats):
            order = args.backends if repeat % 2 == 0 else list(reversed(args.backends))
            for backend in order:
                root = args.out / f'{repeat}-{backend.replace("/", "_")}'
                root.mkdir()
                selected = copy.deepcopy(profile)
                selected['engine']['attention'] = backend
                save(root / 'profile.json', selected)
                cold = [sys.executable, '-m', 'freevideo_engine.cold_pages', '--cache', str(args.cache),
                        '--base', str(base), '--out', str(root / 'cold-pages.json')]
                env = dict(os.environ, **{LOCK_ENV: str(descriptor)})
                if not windows():
                    processes.run(cold, env=env, pass_fds=(descriptor,), check=True)
                for variable, name in [('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'), ('CUTE_DSL_CACHE_DIR', 'cute')]:
                    env[variable] = str(root / 'compiler' / name)
                command = [sys.executable, '-m', 'freevideo_engine', 'generate', '--cache', str(args.cache),
                           '--conditioning', str(args.conditioning), '--profile', str(root / 'profile.json'),
                           '--out', str(root / 'video.mp4'), '--seed', str(args.seed)]
                harness = [sys.executable, '-m', 'freevideo_engine.capacity',
                           '--budget-gb', str(profile['gpu_budget_gb']),
                           '--allocator-config', profile['allocator_config'], '--out', str(root / 'measurement')]
                if args.require_ram_limit:
                    harness += ['--ram-gb', str(profile['inference_ram_budget_gb'])]
                before = time.perf_counter()
                if windows():
                    from .testing import run_process
                    observed = run_process(command, root, env, env.get('CUDA_VISIBLE_DEVICES', '0'),
                        timeout=10800, minimum_ram=2 * 2**30, pass_fds=(descriptor,),
                        maximum_gpu=profile['gpu_budget_gb'] * 1e9,
                        maximum_working_ram=profile['inference_ram_budget_gb'] * 1e9)
                    from .system import memory_complete
                    if observed['status'] == 'complete' and (observed['gpu'].get('gpu_peak_bytes') is None
                            or observed['gpu'].get('sampling_errors') or not memory_complete(observed['ram'])):
                        observed.update(status='measurement_failed', error='Complete GPU/RAM telemetry is required for a valid benchmark.')
                    measurement = root / 'measurement'
                    measurement.mkdir()
                    save(measurement / 'result.json', observed)
                    code, status = observed['returncode'], observed['status']
                else:
                    command = harness + ['--', *command]
                    with (root / 'worker.log').open('w', encoding='utf-8') as log:
                        result = processes.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(descriptor,))
                    code, status = result.returncode, 'complete' if result.returncode == 0 else 'error'
                row = {'backend': backend, 'repeat': repeat, 'status': status,
                       'process_seconds': time.perf_counter() - before, 'exit_code': code, 'output': str(root)}
                metrics = root / 'video.request.json'
                if metrics.exists():
                    row['request'] = json.loads(metrics.read_text(encoding='utf-8'))
                measured = root / 'measurement/result.json'
                if measured.exists():
                    row['measurement'] = json.loads(measured.read_text(encoding='utf-8'))
                report['rows'].append(row)
                save(args.out / 'comparison.json', report)
                print(json.dumps({'event': 'full_request_complete', **row}), flush=True)
        report['success'] = all(r['status'] == 'complete' for r in report['rows'])
    finally:
        save(args.out / 'comparison.json', report)


if __name__ == '__main__':
    main()
