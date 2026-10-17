"""Whole-device Ascend sampling through DCMI, beside the NVML monitor.

The CUDA monitor reads NVML, which an Ascend host does not have. DCMI (the
DaVinci card management interface shipped with the driver) reports the same
whole-device facts, so this module writes the same CSV columns and returns the
same result shape as :class:`freevideo_engine.monitoring.Monitor`. Every value
is the whole device's, other processes included, exactly like the NVML path.

DCMI reports per-process device memory, not per-process utilization, so the
core-share analysis NVML allows is not claimed here: ``other_programs`` is
reported as unsupported and the measured memory is reported separately.
"""
from __future__ import annotations
import csv
import ctypes
import os
from pathlib import Path
import threading
import time

HBM_UNIT = 2 ** 20                      # dcmi_hbm_info sizes are MB
DCMI_OK = 0
MIB = 2 ** 20

# enum dcmi_freq_type
FREQ_HBM = 6
FREQ_AICORE_CURRENT = 7

# DCMI_UTILIZATION_RATE_*
UTIL_AICORE = 2
UTIL_CTRLCPU = 4
UTIL_HBM = 6
UTIL_AIVECTOR = 12
UTIL_NPU = 13

LIBRARY_CANDIDATES = ('libdcmi.so', '/usr/local/dcmi/libdcmi.so',
                      '/usr/local/Ascend/driver/lib64/driver/libdcmi.so')

PROCESS_SLOTS = 256
BUSY_UTIL_THRESHOLD = 50


class HbmInfo(ctypes.Structure):
    # struct dcmi_hbm_info
    _fields_ = [('memory_size', ctypes.c_ulonglong), ('freq', ctypes.c_uint),
                ('memory_usage', ctypes.c_ulonglong), ('temp', ctypes.c_int),
                ('bandwith_util_rate', ctypes.c_uint)]


class ProcMem(ctypes.Structure):
    # struct dcmi_proc_mem_info
    _fields_ = [('proc_id', ctypes.c_int), ('proc_mem_usage', ctypes.c_ulong)]


class DcmiError(RuntimeError):
    pass


def library(candidates=LIBRARY_CANDIDATES):
    """The loaded DCMI library, or None when this host has none."""
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue
        lib.dcmi_get_card_list.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int]
        lib.dcmi_get_device_num_in_card.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.dcmi_get_device_hbm_info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(HbmInfo)]
        lib.dcmi_get_device_utilization_rate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                         ctypes.POINTER(ctypes.c_uint)]
        lib.dcmi_get_device_temperature.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.dcmi_get_device_power_info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.dcmi_get_device_frequency.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_uint)]
        lib.dcmi_get_device_resource_info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ProcMem),
                                                      ctypes.POINTER(ctypes.c_int)]
        initialise = getattr(lib, 'dcmi_init', None)
        if initialise is None or initialise() != DCMI_OK:
            continue
        return lib
    return None


def devices(lib):
    """Every (card, device) the driver reports, in the driver's own order."""
    count = ctypes.c_int(0)
    cards = (ctypes.c_int * 64)()
    if lib.dcmi_get_card_list(ctypes.byref(count), cards, 64) != DCMI_OK:
        raise DcmiError('DCMI could not list cards')
    found = []
    for card in [cards[index] for index in range(max(0, min(count.value, 64)))]:
        number = ctypes.c_int(0)
        if lib.dcmi_get_device_num_in_card(card, ctypes.byref(number)) != DCMI_OK:
            continue
        found += [(card, device) for device in range(max(0, number.value))]
    if not found:
        raise DcmiError('DCMI reported no Ascend device')
    return found


def read(lib, card, device):
    """One whole-device reading; every field is the driver's own, never estimated."""
    hbm = HbmInfo()
    if lib.dcmi_get_device_hbm_info(card, device, ctypes.byref(hbm)) != DCMI_OK:
        raise DcmiError('DCMI HBM read failed for card %d device %d' % (card, device))
    if not hbm.memory_size or hbm.memory_usage > hbm.memory_size:
        raise DcmiError('DCMI returned unavailable device memory; no zero or estimated peak is reported.')
    utilization = ctypes.c_uint(0)
    rc = lib.dcmi_get_device_utilization_rate(card, device, UTIL_AICORE, ctypes.byref(utilization))
    util = int(utilization.value) if rc == DCMI_OK and utilization.value <= 100 else None
    temperature = ctypes.c_int(0)
    rc = lib.dcmi_get_device_temperature(card, device, ctypes.byref(temperature))
    if rc != DCMI_OK:
        temperature = ctypes.c_int(hbm.temp)
    power = ctypes.c_int(0)
    rc = lib.dcmi_get_device_power_info(card, device, ctypes.byref(power))
    # DCMI reports power in tenths of a watt.
    power_mw = power.value * 100 if rc == DCMI_OK and 0 <= power.value < 100000 else None
    core = ctypes.c_uint(0)
    rc = lib.dcmi_get_device_frequency(card, device, FREQ_AICORE_CURRENT, ctypes.byref(core))
    clock_mhz = int(core.value) if rc == DCMI_OK and 0 < core.value < 10000 else None
    return dict(used_bytes=hbm.memory_usage * HBM_UNIT, total_bytes=hbm.memory_size * HBM_UNIT,
                gpu_util_percent=util, temperature_c=int(temperature.value),
                power_mw=power_mw, sm_clock_mhz=clock_mhz,
                memory_clock_mhz=int(hbm.freq) if 0 < hbm.freq < 10000 else None,
                hbm_bandwidth_percent=int(hbm.bandwith_util_rate))


def processes(lib, card, device):
    """{pid: bytes} of the processes holding device memory; None when unreadable.

    The driver leaves a process out of a device it holds nothing on, which is
    also how the same call tells a device apart from its idle siblings.
    """
    rows = (ProcMem * PROCESS_SLOTS)()
    number = ctypes.c_int(0)
    rc = lib.dcmi_get_device_resource_info(card, device, rows, ctypes.byref(number))
    if rc != DCMI_OK or not 0 <= number.value <= PROCESS_SLOTS:
        return None
    # proc_mem_usage is bytes, not the MB npu-smi prints from the same call: a
    # process holding 1143 MiB is reported as 1_198_844_086 here.
    return {rows[index].proc_id: int(rows[index].proc_mem_usage)
            for index in range(number.value) if rows[index].proc_id > 0}


def select(available, device):
    """The device a caller asked for, by index when it can be, else the first."""
    if device is None:
        return available[0]
    text = str(device)
    if text.isdigit():
        index = int(text)
        if not 0 <= index < len(available):
            raise DcmiError('DCMI has no device ' + text)
        return available[index]
    return available[0]


class Gpu:
    """The DCMI counterpart of ``background_room.Gpu``: the same whole-device readings.

    ``busy`` needs per-process utilizations, which DCMI does not report, so it
    answers None exactly as an NVML driver without that call does. Only an index
    or None selects a chip: DCMI cannot look a UUID up, and guessing a chip
    would make a caller's occupancy check wrong rather than absent.
    """

    def __init__(self, device=None, lib=None):
        if device is not None and not str(device).isdigit():
            raise DcmiError('DCMI selects a device by index; a UUID needs a running process to resolve it.')
        self.lib = lib if lib is not None else library()
        if self.lib is None:
            raise DcmiError('DCMI is not installed')
        self.available = devices(self.lib)
        self.device = select(self.available, device)

    def utilization(self):
        value = ctypes.c_uint(0)
        if self.lib.dcmi_get_device_utilization_rate(*self.device, UTIL_AICORE, ctypes.byref(value)) != DCMI_OK:
            return None
        return int(value.value) if value.value <= 100 else None

    def memory(self):
        try:
            values = read(self.lib, *self.device)
        except DcmiError:
            return None
        return values['used_bytes'], values['total_bytes']

    def processes(self):
        return processes(self.lib, *self.device)

    def busy(self, since):
        """None: DCMI reports per-process memory, not per-process device time."""
        return None

    def close(self):
        self.lib = None


def local_family(pid):
    """This monitor's process and, when known, the worker's, as one set."""
    pids = {os.getpid()}
    if pid:
        pids.add(int(pid))
    try:
        from .background_room import family
        found = family(os.getpid())
        if pid:
            found |= family(int(pid))
        return found or pids
    except OSError:
        return pids


class Programs:
    """Other programs' device memory while the worker computes.

    The shape matches the NVML section so readers keep one vocabulary, but DCMI
    has no per-process utilization: ``checks`` stays zero and no core share is
    claimed. What it does report honestly is per-process device memory.
    """
    seconds = 1.
    share_percent = 20

    def __init__(self, lib, monitor, pid=None):
        self.lib, self.monitor, self.pid = lib, monitor, pid
        self.checked = 0.
        self.supported = None
        self.own_reported = False
        self.checks = self.shared = 0
        self.own_total = self.own_peak = 0
        self.other_total = self.other_peak = 0
        self.ours = local_family(pid)
        self.retargets = 0

    def check(self, util, used_bytes):
        now = time.monotonic()
        if now - self.checked < self.seconds:
            return
        self.checked = now
        self.monitor.follow(self.ours)
        found = processes(self.lib, *self.monitor.device)
        if found is None:
            self.supported = False
            return
        self.supported = True
        own = sum(size for pid, size in found.items() if pid in self.ours)
        other = sum(size for pid, size in found.items() if pid not in self.ours)
        self.own_reported = self.own_reported or own > 0
        if not self.own_reported or util is None or util < BUSY_UTIL_THRESHOLD:
            return
        self.checks += 1
        self.shared += bool(used_bytes and other >= self.share_percent / 100. * used_bytes)
        self.own_total += own
        self.own_peak = max(self.own_peak, own)
        self.other_total += other
        self.other_peak = max(self.other_peak, other)

    def result(self):
        # The core-share section keeps the NVML keys and their meaning: DCMI
        # cannot measure a process's share of the cores, so nothing is claimed
        # there. Bytes never enter a `*_percent` field.
        checked = dict(supported=False, own_reported=False, checks=0, shared_checks=0,
                       check_seconds=self.seconds, share_percent=self.share_percent,
                       scope='DCMI reports per-process device memory, not per-process utilization, so no '
                             'share of the device cores is claimed here. Device memory held by every '
                             'process is in other_programs_memory.')
        memory = dict(supported=bool(self.supported), own_reported=self.own_reported, checks=self.checks,
                      shared_checks=self.shared, check_seconds=self.seconds,
                      share_percent=self.share_percent,
                      scope='Device memory held by each process, read once a second through DCMI while the '
                            'worker was computing on a busy device. shared_checks counts readings at or above '
                            'share_percent of the device memory then in use, including this worker\'s.')
        if self.checks:
            memory.update(own_mean_bytes=round(self.own_total / self.checks), own_peak_bytes=self.own_peak,
                          other_mean_bytes=round(self.other_total / self.checks),
                          other_peak_bytes=self.other_peak)
        return checked, memory


class Monitor:
    """The DCMI counterpart of ``monitoring.Monitor``: same CSV, same result.

    ``device`` accepts the CUDA-style identity the caller already has (the NPU's
    UUID, an index, or None). DCMI cannot look a UUID up, so the device is
    confirmed by looking for this monitor's and the worker's own processes: the
    first reading that finds them decides, and the baseline starts from there.
    """
    def __init__(self, path, interval=0.05, device=None, pid=None, lib=None):
        self.lib = lib if lib is not None else library()
        if self.lib is None:
            raise DcmiError('DCMI is not installed; no whole-device sampling is available.')
        self.available = devices(self.lib)
        self.device = self.select(device)
        self.requested = device
        self.confirmed = False
        self.retargeted = 0
        self.pid = int(pid) if pid else None
        self.programs = Programs(self.lib, self, pid)
        self.path, self.interval = Path(path), interval
        self.stop_event = threading.Event()
        self.peak = self.baseline = self.last = 0
        self.count = 0
        self.errors = []
        self.max_temperature = None
        self.max_power_mw = None
        from .monitoring import Activity
        self.activity, self.busy_activity = Activity(), Activity()

    def select(self, device):
        return select(self.available, device)

    def follow(self, ours):
        """Retarget to the device that actually holds our processes, once.

        The worker allocates on exactly one chip; its first appearance there is
        the driver's own answer to which one, and it is the only answer DCMI
        offers for a UUID. The peak restarts from that device's current use, so
        the other chip's numbers are never reported as this request's.
        """
        if self.confirmed:
            return
        for card, device in self.available:
            found = processes(self.lib, card, device)
            if found and any(pid in found for pid in ours):
                if (card, device) != self.device:
                    self.device = (card, device)
                    self.retargeted += 1
                    if self.count:
                        self.baseline = self.peak = read(self.lib, card, device)['used_bytes']
                self.confirmed = True
                return

    def sample(self):
        values = read(self.lib, *self.device)
        auxiliary = [values['temperature_c'], values['power_mw'], values['sm_clock_mhz'],
                     values['memory_clock_mhz']]
        return values, auxiliary

    def start(self):
        values, _ = self.sample()
        self.baseline = self.peak = self.last = values['used_bytes']
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def loop(self):
        with self.path.open('w', buffering=1, encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch_seconds', 'monotonic_seconds', 'used_bytes', 'total_bytes',
                             'gpu_util_percent', 'temperature_c', 'power_mw', 'sm_clock_mhz',
                             'memory_clock_mhz', 'clock_reason_mask'])
            while not self.stop_event.is_set():
                try:
                    values, auxiliary = self.sample()
                    used = values['used_bytes']
                    self.peak = max(self.peak, used)
                    self.last = used
                    self.count += 1
                    self.activity.add(values['gpu_util_percent'], auxiliary, None)
                    if values['gpu_util_percent'] is not None and values['gpu_util_percent'] >= BUSY_UTIL_THRESHOLD:
                        self.busy_activity.add(values['gpu_util_percent'], auxiliary, None)
                    if auxiliary[0] is not None:
                        self.max_temperature = max(self.max_temperature or 0, auxiliary[0])
                    if auxiliary[1] is not None:
                        self.max_power_mw = max(self.max_power_mw or 0, auxiliary[1])
                    writer.writerow([time.time(), time.monotonic(), used, values['total_bytes'],
                                     values['gpu_util_percent'], *auxiliary, None])
                    self.programs.check(values['gpu_util_percent'], used)
                except Exception as error:
                    if str(error) not in self.errors and len(self.errors) < 20:
                        self.errors.append(str(error))
                self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        checked, memory = self.programs.result()
        return {'device': 'card%d/device%d' % self.device, 'device_requested': str(self.requested),
                'device_confirmed_by_process': self.confirmed, 'device_retargets': self.retargeted,
                'gpu_peak_bytes': self.peak, 'gpu_peak_mib': self.peak / MIB,
                'gpu_baseline_bytes': self.baseline, 'gpu_last_bytes': self.last,
                'max_temperature_c': self.max_temperature, 'max_power_mw': self.max_power_mw,
                'activity': self.activity.result(), 'busy_activity': self.busy_activity.result(),
                'busy_util_threshold_percent': BUSY_UTIL_THRESHOLD,
                'activity_scope': 'Whole device during this worker request, including load and decode. '
                                  'Sample means, not duration-weighted measurements. Busy subset has AICore utilization >= 50%. '
                                  'DCMI reports no clock-reason counters, so no slowdown is attributed.',
                'other_programs': checked, 'other_programs_memory': memory,
                'other_programs_scope': 'Per-process device memory read once a second through DCMI while the worker was '
                                        'computing on a busy device: the memory held by this worker and its monitor, and by '
                                        'processes outside them. DCMI reports no per-process utilization, so core shares are '
                                        'not claimed.',
                'sampling_interval_seconds': self.interval, 'samples': self.count,
                'sampling_errors': self.errors, 'sample_csv': str(self.path),
                'sampler': 'dcmi'}


def monitor(path, interval=0.05, device=None, pid=None):
    """A started DCMI monitor, or None when this host has no DCMI to sample through."""
    try:
        return Monitor(path, interval=interval, device=device, pid=pid).start()
    except (DcmiError, OSError, ValueError):
        return None


def memory_budget(device=None):
    """Whole-device memory in use by every process, read on demand."""
    lib = library()
    if lib is None:
        raise DcmiError('DCMI is not installed')
    card, chip = Monitor.__new__(Monitor).select.__get__(type('_S', (), {'available': devices(lib)})(), device)
    return lib, (card, chip)
