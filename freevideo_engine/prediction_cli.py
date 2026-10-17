"""Preview local runtime/resource forecasts without starting inference or probes.

The parent stays torch-free. Hardware detection uses the existing short-lived
read-only detector; this command never runs model workers or kernel probes.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .adaptive import attempt_geometry, benchmark_allocator_limit, history_path, local_identity
from .geometry import geometry
# detect_hardware picks the detector by installation; imported under the name the rest
# of this file already calls.
from .cli import detect_hardware as detect
from .kernel_capabilities import available_backends
from .paths import data_root
from .policy import choose
from .prediction import predict
from .prediction_policy import select
from .prediction_ui import show
from .resource_history import ResourceHistory, ResourceHistoryError
from .tuning import digest
from . import tuning


def _emit(value, stream):
    output = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    encoding = getattr(stream, 'encoding', None)
    if encoding:
        try:
            output.encode(encoding)
        except UnicodeEncodeError:
            output = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)
    print(output, file=stream)


def _selected_cache(explicit):
    if explicit is not None:
        return explicit.expanduser()
    machine = data_root() / 'machine.json'
    if not machine.is_file():
        return None
    value = json.loads(machine.read_text(encoding='utf-8'))
    return Path(value['cache']).expanduser() if value.get('cache') else None


def _conditioning(path, observations):
    if path is None:
        return {'sha256': None, 'text_tokens': None, 'source': 'unknown input token length'}
    checksum = digest(path.expanduser())
    counts = set()
    for row in observations:
        observed = row.get('observation') or {}
        if observed.get('conditioning_sha256') != checksum:
            continue
        count = observed.get('text_tokens')
        if count is None:
            count = (observed.get('geometry') or {}).get('text_tokens')
        if count is None:
            count = (row.get('geometry') or {}).get('text_tokens')
        if type(count) is int and count >= 0:
            counts.add(count)
    count = next(iter(counts)) if len(counts) == 1 else None
    source = ('matching conditioning SHA256 in complete local history' if count is not None
              else 'conflicting historical token lengths' if len(counts) > 1
              else 'conditioning SHA256 has no matching token observation')
    return {'sha256': checksum, 'text_tokens': count, 'source': source}


def _tuned_profile(profile, cache, hardware, canvas, tokens, enabled):
    decision = {'profile_id': None, 'input_tokens_known': tokens is not None}
    if not enabled:
        decision['reason'] = 'Saved tuning bypassed by explicit profile, backend or --no-tuning'
        return profile, decision
    if tokens is None:
        decision['reason'] = ('Initial automatic policy: input token length is unknown. '
                              'Generation may apply validated tuning after text encoding.')
        return profile, decision
    if cache is None or not tuning.state_path().is_file():
        decision['reason'] = 'No saved local tuning available for this preview'
        return profile, decision
    try:
        machine = json.loads((data_root() / 'machine.json').read_text(encoding='utf-8'))
        if cache.resolve() != Path(machine['cache']).resolve():
            decision['reason'] = 'Prepared cache differs from the optimized installation'
            return profile, decision
        identity = tuning.identity(machine, hardware)
        state, reason = tuning.load_state(identity)
        if state:
            profile, applied = tuning.apply_profile(state, profile, canvas, tokens)
            decision.update(profile_id=applied, reason=('Apply the locally validated tuning profile' if applied
                else 'No validated tuning profile matches this input, geometry and current resources'))
        else:
            decision['reason'] = reason
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        decision['reason'] = 'Could not validate saved tuning: ' + str(error)
    return profile, decision


def main(argv=None):
    parser = argparse.ArgumentParser(prog='freevideo predict',
        description='Preview phase times and memory from complete local history; no generation or kernel probes.')
    parser.add_argument('--history', type=Path, help='Attempt database; defaults to this installation')
    parser.add_argument('--cache', type=Path, help='Prepared model cache; defaults to the setup machine configuration')
    parser.add_argument('--base', type=Path, help='Same base model path as the intended generation')
    parser.add_argument('--checkpoint', type=Path, help='Same checkpoint path as the intended generation')
    parser.add_argument('--profile', type=Path, help='Forecast an explicit profile without changing its placement')
    parser.add_argument('--conditioning', type=Path, help='Optional existing input: look up its token length by SHA256, without loading tensors')
    parser.add_argument('--width', type=int, default=1344)
    parser.add_argument('--height', type=int, default=768)
    length = parser.add_mutually_exclusive_group()
    length.add_argument('--frames', type=int)
    length.add_argument('--seconds', type=float)
    parser.add_argument('--task', choices=('t2va', 'i2va', 'l2va', 'fl2va', 'ref2va', 'ref2va_audio', 'ref2va_av'), help='Override the profile task (default: t2va)')
    parser.add_argument('--cache-state', choices=('cold', 'warm', 'unknown'), default='unknown',
                        help='Explicit AdaLN cache regime for load/worker time; never inferred from history')
    parser.add_argument('--allocator-limit-gib', type=float,
                        help='Forecast the same explicit PyTorch benchmark cap; capped and ordinary history stay separate')
    parser.add_argument('--attention', default='auto', help='Requested backend for automatic profiles; never probes kernels')
    parser.add_argument('--vram-gib', type=float)
    parser.add_argument('--ram-gib', type=float)
    parser.add_argument('--gpu-reserve-gib', type=float)
    parser.add_argument('--ram-reserve-gib', type=float)
    parser.add_argument('--no-tuning', action='store_true', help='Preview without saved numerical tuning, matching generate --no-tuning')
    parser.add_argument('--no-history-placement', action='store_true', help='Preview current policy without history placement selection')
    parser.add_argument('--json', dest='json_output', action='store_true', help='Print complete machine-readable forecast and selection evidence')
    args = parser.parse_args(argv)
    try:
        canvas = geometry(args.width, args.height, frames=args.frames, seconds=args.seconds)
        from .two_pass import render_canvas
        canvas = render_canvas(canvas)[0]
        hardware = detect()
        if args.profile:
            profile = json.loads(args.profile.expanduser().read_text(encoding='utf-8'))
            if 'gpu_budget_bytes' in profile:
                cap = profile.get('benchmark_allocator_limit_bytes')
                profile = dict(gpu_budget_gb=profile['gpu_budget_bytes'] / 1e9,
                    inference_ram_budget_gb=profile['ram_budget_bytes'] / 1e9,
                    engine=profile['engine'], decoder=profile['decoder'],
                    allocator_config=profile['allocator_config'], policy=profile)
                if cap is not None:
                    profile['benchmark_allocator_limit_bytes'] = cap
        else:
            profile = choose(hardware, **{name: getattr(args, name) for name in
                ('vram_gib', 'ram_gib', 'attention', 'gpu_reserve_gib', 'ram_reserve_gib')},
                available_backends=available_backends(hardware, probe_missing=False), canvas=canvas).legacy_profile()
        profile = copy.deepcopy(profile)
        for name in ('base', 'checkpoint'):
            value = getattr(args, name)
            if value is not None:
                profile['engine'][name] = str(value.expanduser().resolve())
        if args.task:
            profile['engine']['task'] = args.task
        limit = benchmark_allocator_limit(profile, args.allocator_limit_gib, hardware)
        if limit is not None:
            profile['benchmark_allocator_limit_bytes'] = limit
        context = {'adaln_cache_state': args.cache_state}
        profile['allocator_config'] = profile.get('allocator_config', os.environ.get('PYTORCH_ALLOC_CONF',
            os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'backend:native,pinned_max_round_threshold_mb:128')))
        cache = _selected_cache(args.cache)
        identity = local_identity(hardware, cache) if cache is not None else None
        selected_history = args.history.expanduser() if args.history else history_path()
        history = ResourceHistory(selected_history) if selected_history.exists() else None
        observations = (history.observations(identity=identity, limit=512, include_details=False)
                        if history is not None and identity is not None else [])
        conditioning = _conditioning(args.conditioning, observations)
        profile, tuning_decision = _tuned_profile(profile, cache, hardware, canvas, conditioning['text_tokens'],
            not args.profile and not args.no_tuning and args.attention == 'auto')
        # No model identity is guessed when setup has not supplied a cache.
        forecast_identity = identity if identity is not None else {'unverified_model_identity': True}
        if args.profile or args.no_history_placement:
            decision = {'changed': False, 'reason': 'Explicit profile or --no-history-placement: preview preserves placement and backend'}
        elif observations:
            profile, decision = select(profile, canvas, hardware, observations, forecast_identity,
                text_tokens=conditioning['text_tokens'], execution_context=context)
        else:
            decision = {'changed': False, 'reason': 'No compatible complete history; show the current automatic policy'}
        compatibility = None
        if identity is not None:
            from .compatibility import Store, apply as apply_compatibility
            profile, compatibility = apply_compatibility(profile, Store(selected_history.parent).status(identity))
        forecast = predict(observations, forecast_identity, profile, canvas, text_tokens=conditioning['text_tokens'],
                           execution_context=context)
        forecast['tuning_preview'] = tuning_decision
        forecast['reasons'].append(tuning_decision['reason'])
        if identity is None:
            forecast['reasons'].append('No prepared model cache selected; run setup or supply --cache for exact local model/software identity.')
        result = {'forecast': forecast, 'profile': profile, 'selection': decision,
                  'compatibility': compatibility,
                  'history': str(selected_history), 'history_exists': history is not None,
                  'model_identity_verified': identity is not None, 'conditioning': conditioning, 'tuning': tuning_decision,
                  'scope': 'Preview only; video-engine timings exclude text encoding. No inference or kernel probe was started.'}
        if args.json_output:
            _emit(result, sys.stdout)
        else:
            show(forecast)
            if identity is None:
                print('Select a prepared cache with --cache or complete setup to match local history.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ResourceHistoryError) as error:
        _emit({'error': str(error), 'inference_started': False, 'kernel_probes_started': False}, sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
