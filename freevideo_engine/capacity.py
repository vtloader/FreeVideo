#!/usr/bin/env python3
"""Measure an isolated GPU experiment natively or with optional enforced limits."""
from .torch_device import device_module, device_name
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freevideo_engine.monitoring import Monitor, save
from freevideo_engine.locking import runtime_lock, device_lock_path, LOCK_ENV


# A harness stop is a known controller stop. Left pending, the attempt looks
# like an abandoned GPU crash to the next request on that device, which raises
# a compatibility incident and caps every later request there to the smallest
# work groups. optimize.py settles its own stops for exactly this reason.
CONTROLLED_STOPS = ('whole_gpu_budget_exceeded', 'timeout', 'ram_cgroup_oom',
                    'monitor_failed', 'harness_error')


def settle_controlled_stop(reason, since, device):
    """Settle attempts this run left pending, so they are not read as crashes."""
    try:
        from freevideo_engine.paths import data_root
        from freevideo_engine.resource_history import ResourceHistory
        path = data_root() / 'resource-history.sqlite3'
        if not path.is_file():
            return []
        history = ResourceHistory(path)
        settled = []
        for attempt in history.attempts():
            # The device lease is held exclusively, so a pending attempt
            # started after this run began belongs to this run.
            if attempt.get('outcome') != 'pending' or (attempt.get('started') or 0) < since:
                continue
            settled.append(history.finish(attempt['id'], 'cancelled', details=dict(
                reason='Capacity harness stopped this attempt: ' + str(reason),
                controlled_worker_stop=True, capacity_device=str(device))))
        return settled
    except Exception as error:
        # This runs while the measurement is being written out; a settlement
        # problem is recorded, never allowed to replace the result.
        return [dict(error='%s: %s' % (type(error).__name__, error))]


def mps_client_device(pipe, log, device):
    """Which GPU an MPS client of this daemon actually gets.

    An MPS client does not choose its device: CUDA_VISIBLE_DEVICES applies to
    the control daemon, and clients receive the device set the daemon was
    started with. Attaching several measurements for different devices to one
    shared daemon therefore sends all of them to the same GPU, silently. Ask a
    throwaway client which device it received and compare.
    """
    code = ('import torch;'
            'p = device_module(torch).get_device_properties(torch.cuda.current_device());'
            'u = str(getattr(p, "uuid", ""));'
            'print(u if u.startswith(("GPU-", "MIG-")) else "GPU-" + u)')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device),
                       CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(log))
    result = subprocess.run([sys.executable, '-c', code], env=environment,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('An MPS client of ' + str(pipe) + ' could not start: '
                           + (result.stderr or '')[-400:])
    return result.stdout.strip().splitlines()[-1]

def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-gb', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=10800)
    parser.add_argument('--allocator-config')
    parser.add_argument('--ram-gb', type=float, help='Require a dedicated cgroup hard physical RAM limit (zero swap by default)')
    parser.add_argument('--swap-gb', type=float, default=0, help='Explicit separate swap allowance in decimal GB; requires --ram-gb and a preconfigured bounded cgroup')
    parser.add_argument('--no-mps', action='store_true', help='Use native CUDA; monitor the whole GPU without imposing an MPS client limit')
    parser.add_argument('--mps-pipe', type=Path,
                        help='Attach to an MPS control daemon started outside this run. One daemon '
                             'per user per host serves every device, so measuring several devices at '
                             'once requires sharing it; the per-client memory limit stays per device.')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('The MPS/cgroup capacity harness requires Linux. On Windows use test or bench --mode full for observed memory.')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.budget_gb < 2:
        parser.error('A command and at least 2 GB are required')
    if not math.isfinite(args.swap_gb) or args.swap_gb < 0 or (args.swap_gb and args.ram_gb is None):
        parser.error('--swap-gb must be finite and nonnegative, and requires --ram-gb')
    if args.mps_pipe is not None and args.no_mps:
        parser.error('--mps-pipe attaches to a daemon; it cannot be combined with --no-mps')
    if os.environ.get(LOCK_ENV):
        # A caller already granted a lease; reuse it exactly as before.
        with runtime_lock() as descriptor:
            return measure(args, command, descriptor)
    # Otherwise own the leases directly. The device is chosen the same way the
    # NVML monitor chooses it, before any CUDA context exists. Hold the machine
    # lease shared, so setup and whole-host tests still exclude us, and this
    # device exclusively, so two measurements never share a GPU while the
    # other devices on the host stay usable.
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    lease = device_lock_path(device)
    with runtime_lock(shared=True, inherit=False), \
            runtime_lock(lease, inherit=False) as descriptor:
        # The worker validates its inherited descriptor against this path.
        os.environ['FREEVIDEO_LOCK_PATH'] = str(lease)
        return measure(args, command, descriptor)


def measure(args, command, descriptor):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from freevideo_engine.ram import CgroupMemory, ProcessMemory
    observation = ProcessMemory()
    ram = None
    if args.ram_gb is not None:
        if Path('/proc/self/cgroup').read_text(encoding='utf-8').strip() != '0::/':
            raise RuntimeError('RAM harness requires a dedicated cgroup namespace/container')
        ram = CgroupMemory(int(args.ram_gb * 1e9), swap_budget_bytes=int(getattr(args, 'swap_gb', 0) * 1e9))
    # Resolved before the occupancy check, which tolerates this daemon's server.
    shared_pipe = args.mps_pipe.resolve() if args.mps_pipe is not None else None
    if shared_pipe is not None and not shared_pipe.is_dir():
        raise RuntimeError('MPS pipe directory does not exist: ' + str(shared_pipe))
    monitor = Monitor(root / 'gpu.csv')
    # Occupancy is scoped to the device being measured. A machine-wide check
    # makes this harness unusable on any multi-GPU host, including one
    # measuring several independent budgets on separate devices at once.
    identity = str(monitor.device)
    if not identity.startswith('GPU-'):
        identity = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader', '-i', identity],
            text=True).strip()
    listed = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid',
         '--format=csv,noheader,nounits'], text=True)
    rows = [[field.strip() for field in row.split(',')] for row in listed.splitlines() if row.strip()]
    # Without a usable UUID fall back to the whole host, rather than admit a
    # measurement that another process could be perturbing unseen.
    scoped = identity.startswith('GPU-')
    here = [row for row in rows if row[-1] == identity] if scoped else rows
    # The caller's own MPS server is expected when several devices share one
    # control daemon; it reserves device memory before our client starts.
    # Anything else on this device is still another workload and still refuses.
    ours, foreign, reserved = [], [], 0
    for row in here:
        if shared_pipe is not None and Path(row[1]).name == 'nvidia-cuda-mps-server':
            ours.append(row)
            try:
                reserved += int(float(row[2])) * 2**20
            except (ValueError, IndexError):
                pass
        else:
            foreign.append(row)
    # NVML v1 includes driver-reserved memory even when nvidia-smi has no clients.
    observed = max(0, monitor.sample()[0] - reserved)
    if foreign or observed > 1024 * 2**20:
        raise RuntimeError('Device %s is occupied; stop other GPU work on it before measuring'
                           % (identity if scoped else monitor.device))
    plan_identity = identity if scoped else None
    plan_scope = 'device' if scoped else 'host'
    if shared_pipe is not None and scoped:
        received = mps_client_device(shared_pipe, root / 'mps', monitor.device)
        if received != identity:
            raise RuntimeError(
                'The shared MPS daemon at %s places clients on %s, not the requested %s. '
                'An MPS client cannot choose its device; the daemon decides. Start one daemon '
                'per device set, or measure this device with --no-mps.'
                % (shared_pipe, received, identity))
    plan_tolerated = dict(mps_server_processes=len(ours), mps_server_reserved_bytes=reserved,
                          observed_before_start_bytes=observed)
    budget = int(args.budget_gb * 1_000_000_000)
    # Leave room for server/driver allocations outside a client's accounting.
    # One GiB covers a daemon serving a single device. A daemon shared across
    # devices costs more per device: an 8 GiB client under a shared daemon
    # reached a 9.46 GiB whole-device peak against a 9.00 GiB budget on an
    # H200, so reserve two GiB there instead of aborting on the server's own
    # growth. The client limit, not this reserve, is what bounds the engine.
    server_reserve = (2048 if shared_pipe is not None else 1024) * 2**20
    client_mib = (budget - server_reserve) // 2**20
    # PIDs can repeat across containers and after restarts. Never reuse an old
    # daemon's socket directory; native CUDA needs no MPS directory at all.
    # A shared daemon is supplied by the caller, which also owns its lifetime.
    pipe = shared_pipe or (Path(tempfile.mkdtemp(prefix='freevideo-mps-')) if not args.no_mps else None)
    (root / 'mps').mkdir()
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=monitor.device,
               CUDA_MPS_PIPE_DIRECTORY=str(pipe),
               CUDA_MPS_LOG_DIRECTORY=str(root / 'mps'),
               CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=f'0={client_mib}M',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONUNBUFFERED='1')
    env[LOCK_ENV] = str(descriptor)
    if args.no_mps:
        for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'):
            env.pop(key, None)
    if args.allocator_config is not None:
        env['PYTORCH_ALLOC_CONF'] = args.allocator_config
        env['PYTORCH_CUDA_ALLOC_CONF'] = args.allocator_config
    plan = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'selected_gpu': monitor.device, 'selected_gpu_uuid': plan_identity,
            'occupancy_check_scope': plan_scope, 'occupancy_tolerated': plan_tolerated,
            'mps_pipe_directory': str(pipe) if pipe else None,
            'budget_bytes': budget, 'budget_unit': 'decimal GB',
            'mps_client_limit_mib': None if args.no_mps else client_mib, 'command': command,
            'mps_server_reserve_bytes': None if args.no_mps else server_reserve,
            'mps_daemon': 'none' if args.no_mps else 'shared' if shared_pipe else 'own',
            'gpu_limit_mode': 'NVML monitor, native CUDA' if args.no_mps else
                              'MPS client limit and whole-GPU NVML monitor',
            'allocator_config': env.get('PYTORCH_ALLOC_CONF', env.get('PYTORCH_CUDA_ALLOC_CONF', 'default')),
            'compiler_cache_environment': {key: env[key] for key in
                ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUTE_DSL_CACHE_DIR', 'XDG_CACHE_HOME') if key in env},
            'diagnostic_environment': {key: env[key] for key in ('CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG') if key in env},
            'hardware_simulation': 'Memory capacity only; GPU compute and bandwidth unchanged',
            'timeout_seconds': args.timeout,
            'ram_limit_enforced': ram is not None,
            'execution_environment': 'container' if Path('/.dockerenv').exists() else 'host',
            'scope': 'Process start through exit, including imports, load, generate, decode and save'}
    if ram is not None:
        plan.update(ram_budget_bytes=ram.budget_bytes, ram_limit_enforced=True,
                    swap_budget_bytes=ram.swap_budget_bytes,
                    memory_mode='bounded_swap' if ram.swap_budget_bytes else 'no_swap',
                    ram_cgroup_initial=ram.initial)
    repo = Path(__file__).resolve().parents[1]
    source_files = sorted((repo / 'freevideo_engine').glob('*.py')) + sorted((repo / 'scripts').glob('*.py'))
    source_files += sorted((repo / 'scripts').glob('*.sh'))
    source_files += sorted((repo / 'tests').glob('*.py'))
    source_files += sorted((repo / 'comfy_nodes').rglob('*.py')) + sorted((repo / 'profiles').rglob('*.json'))
    source_files += sorted((repo / 'freevideo_engine').glob('*.json'))
    plan['source_sha256'] = {}
    for path in source_files:
        relative = path.relative_to(repo)
        data = path.read_bytes()
        plan['source_sha256'][str(relative)] = hashlib.sha256(data).hexdigest()
        destination = root / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    plan['engine_source_snapshot'] = str(root / 'source')
    # Python otherwise prepends the working/script directory ahead of PYTHONPATH.
    # -P semantics ensure imports really use the frozen experiment source.
    env['PYTHONSAFEPATH'] = '1'
    env['PYTHONPATH'] = os.pathsep.join((str(root / 'source'), str(root / 'source/setup'), env.get('PYTHONPATH', '')))
    plan['python_safe_path'] = True
    if len(command) > 1 and command[1].endswith('.py'):
        script = Path(command[1]).resolve()
        if script.is_relative_to(repo):
            snapshot = root / 'source' / script.relative_to(repo)
            if snapshot.is_file():
                plan['requested_command'] = list(command)
                command = [command[0], str(snapshot), *command[2:]]
                plan['command'] = command
    save(root / 'plan.json', plan)
    daemon_started = False
    proc = None
    result = dict(plan, status='starting')
    cpu_peak_rss = 0
    disk_read_bytes = 0
    ram_log = (root / 'ram.csv').open('w', buffering=1, encoding='utf-8')
    ram_writer = csv.writer(ram_log)
    ram_writer.writerow(['elapsed_seconds', 'tree_rss_bytes', 'tree_pss_bytes', 'system_available_bytes', 'processes'])
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Capacity measurement interrupted by signal {signum}')
    previous_signal = signal.signal(signal.SIGTERM, interrupted)
    monitor.start()
    try:
        if not args.no_mps and shared_pipe is None:
            daemon = subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env,
                                    capture_output=True, text=True, timeout=20)
            (root / 'mps-start.log').write_text(daemon.stdout + daemon.stderr, encoding='utf-8')
            daemon.check_returncode()
            daemon_started = True
        started = time.monotonic()
        wall_start = time.time()
        with (root / 'worker.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, pass_fds=(descriptor,))
        save(root / 'process.json', {'pid': proc.pid})
        reason = None
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            ram_state = ram.snapshot() if ram is not None else None
            memory_observed = observation.sample(proc.pid)
            ram_writer.writerow([elapsed, memory_observed['rss_bytes'], memory_observed['pss_bytes'],
                                 memory_observed['system_available_bytes'], memory_observed['processes']])
            try:
                process_status = (Path('/proc') / str(proc.pid) / 'status').read_text(encoding='utf-8')
                rss = next(int(line.split()[1]) * 1024 for line in process_status.splitlines() if line.startswith('VmRSS:'))
                cpu_peak_rss = max(cpu_peak_rss, rss)
                io = (Path('/proc') / str(proc.pid) / 'io').read_text(encoding='utf-8')
                disk_read_bytes = max(disk_read_bytes, next(int(line.split()[1]) for line in io.splitlines() if line.startswith('read_bytes:')))
            except (OSError, StopIteration):
                pass
            if monitor.errors:
                reason = 'monitor_failed'
            elif monitor.peak > budget:
                reason = 'whole_gpu_budget_exceeded'
            elif ram_state and ram.events_since_start(ram_state).get('oom_kill', 0):
                reason = 'ram_cgroup_oom'
            elif elapsed > args.timeout:
                reason = 'timeout'
            if reason:
                stop_process(proc)
                break
            save(root / 'status.json', {'event': 'running', 'elapsed_seconds': elapsed,
                                      'gpu_used_bytes': monitor.last,
                                      'gpu_peak_bytes': monitor.peak,
                                      'worker_peak_rss_bytes': cpu_peak_rss,
                                      'worker_disk_read_bytes': disk_read_bytes,
                                      'ram_pss_peak_bytes': observation.peak_pss,
                                      'ram_tree_rss_peak_bytes': observation.peak_rss,
                                      'ram_observed': memory_observed,
                                      'ram_cgroup': ram_state})
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        result.update(exit_code=proc.returncode, process_seconds=time.monotonic() - started,
                      status=reason or ('success' if proc.returncode == 0 else 'error'))
    except BaseException as exc:
        result.update(status='harness_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if proc is not None:
            stop_process(proc)
        # Children can start their own sessions. A crashed wrapper must not
        # leave its known server/worker alive with our inherited benchmark lock.
        for (pid, start_tick) in observation.process_io:
            try:
                fields = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8').rsplit(')', 1)[1].split()
                if fields[19] == start_tick and fields[0] != 'Z':
                    os.kill(pid, signal.SIGTERM)
            except (OSError, IndexError):
                pass
        if daemon_started:
            try:
                stopped = subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', env=env,
                                         capture_output=True, text=True, timeout=20)
                (root / 'mps-stop.log').write_text(stopped.stdout + stopped.stderr, encoding='utf-8')
                if stopped.returncode:
                    result['cleanup_error'] = 'MPS shutdown returned ' + str(stopped.returncode)
            except subprocess.TimeoutExpired as error:
                result['cleanup_error'] = str(error)
            if result.get('cleanup_error') and result.get('status') == 'success':
                result['status'] = 'cleanup_error'
        result.update(monitor.stop())
        ram_log.close()
        result.update(observation.result())
        result.update(worker_peak_rss_bytes=cpu_peak_rss, worker_disk_read_bytes=disk_read_bytes,
                      worker_memory_scope='Main worker RSS includes file-backed pages; compiler child processes excluded')
        result['within_budget'] = result['gpu_peak_bytes'] <= budget
        if ram is not None:
            result.update(ram.result())
            result['within_budget'] = result['within_budget'] and result['ram_within_budget']
            if not result['ram_within_budget'] and result['status'] == 'success':
                result['status'] = 'ram_budget_failed'
        if result.get('status') in CONTROLLED_STOPS:
            result['settled_attempts'] = settle_controlled_stop(
                result['status'], locals().get('wall_start', 0), monitor.device)
        result['attempt_completed_successfully'] = result['status'] == 'success'
        result['ram_limit_attempt_passed'] = bool(ram is not None and result['status'] == 'success'
                                                  and result['within_budget'])
        save(root / 'result.json', result)
        save(root / 'status.json', dict(result, event='finished'))
        print(json.dumps(result, indent=2), flush=True)
        signal.signal(signal.SIGTERM, previous_signal)
    return 0 if result['status'] == 'success' and result['within_budget'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
