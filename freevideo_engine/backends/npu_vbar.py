"""Ascend VBAR weight residency driven by comfy-aimdo.

comfy-aimdo reserves a virtual address range (a VBAR) per model and faults weight
pages in on demand, evicting the lower-priority pages when the device is full. The
engine's layer loop is the natural driver: fault a layer before its forward,
unpin it after the forward, and let the library decide what stays resident.

What the channel buys is granularity rather than capacity: the measured placement
keeps 42 of 50 blocks resident and streams the remaining 8, which
``NPULayerResidency``'s host copies fit as well. Paging at 32 MiB lets aimdo hold a
fraction of a layer where whole-layer residency must hold or stream all of it, and
it measured 32.03 s per step against that fallback's 32.65 s. Callers fall back
when aimdo is missing, refuses to initialize, or cannot fault a page on this device
(see ``available``); the fallback is the same arithmetic and copies each offloaded
layer once per step.

A layer's tensors are re-pointed at the VBAR only once every slice of that layer
is ready, so a partly faulted layer is never executed. A slice that cannot be
faulted uses aimdo's documented fallback: a temporary device tensor for that one
forward, released afterwards.
"""
import importlib
import os
import sys
import time
from pathlib import Path

import torch

# The development checkout is the fallback for a machine that has no installed
# package; an installation names its own location through FREEVIDEO_AIMDO_PATH.
DEFAULT_AIMDO_PATH = '/workspace/comfy-aimdo'


def aimdo_search_path(explicit=None):
    """Where to look for a comfy-aimdo checkout that is not importable as a package."""
    return explicit or os.environ.get('FREEVIDEO_AIMDO_PATH') or DEFAULT_AIMDO_PATH


def aimdo_module(search_path=None):
    """Import comfy-aimdo.control, adding a checkout to sys.path when it is not installed."""
    search_path = aimdo_search_path(search_path)
    try:
        return importlib.import_module('comfy_aimdo.control')
    except ImportError:
        candidate = Path(search_path)
        if not (candidate / 'comfy_aimdo' / 'control.py').is_file():
            raise
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return importlib.import_module('comfy_aimdo.control')


# One VBAR page. The fault is the step that fails on hardware the library and the
# device context both initialize on, so the probe has to reach it.
PROBE_BYTES = 32 << 20


def available(search_path=None):
    """Whether aimdo exists here, initializes, and can fault a page on this device.

    Loading the library and initializing the device context are both enough to
    succeed on a die that then refuses the mapping: a VBAR faults its pages through
    the CANN driver, and on the second die of a dual-die package rtMemSetAccess
    fails with drvRetCode 42 after init_device returned True. That failure surfaces
    on the first faulted layer, in the middle of a forward, where no caller can fall
    back -- the offloader's try/except covers construction only. One page is faulted
    here instead, so the choice is settled before a model is placed.

    The device context this initializes is the one the real VBAR then reuses, so the
    probe costs a page rather than a second context.
    """
    try:
        control = aimdo_module(search_path)
        if not control.init('npu'):
            return False
        import torch
        import comfy_aimdo.torch  # noqa: F401  (model_vbar reads control.lib at import)
        from comfy_aimdo.model_vbar import ModelVBAR
        index = torch.npu.current_device()
        if not control.devctxs and not control.init_device(index):
            return False
        vbar = ModelVBAR(PROBE_BYTES, device=index)
        alloc = vbar.alloc(PROBE_BYTES)
        vbar.fault(alloc[1], alloc[2])
        del vbar
        return True
    except Exception as error:
        # Say why. A silent fallback would hide that this device cannot use the
        # faster residency, and the cause is a driver report that nothing downstream
        # can reconstruct.
        print('comfy-aimdo VBAR unusable on this device, using host copies: %s' % error,
              file=sys.stderr, flush=True)
        return False


class VBARLayerResidency:
    """One VBAR holding every layer's weights, faulted per forward."""

    def __init__(self, layers, device='npu', prefetch=True, manage_hooks=True,
                 weight_source=None, watermark_bytes=None, slack_bytes=256 << 20,
                 search_path=None):
        if torch.is_grad_enabled():
            raise RuntimeError('NPU VBAR residency requires gradients disabled')
        if weight_source is not None:
            raise NotImplementedError('NPU VBAR residency keeps host copies; streamed weight sources are not integrated')
        self.control = aimdo_module(search_path)
        if not self.control.init('npu'):
            raise RuntimeError('comfy-aimdo did not initialize its Ascend NPU backend')
        import comfy_aimdo.torch as aimdo_torch
        from comfy_aimdo.model_vbar import (ModelVBAR, vbar_fault,
                                            vbar_signature_compare, vbar_unpin)
        self.bridge = aimdo_torch
        self.fault, self.unpin, self.compare = vbar_fault, vbar_unpin, vbar_signature_compare
        self.layers = list(layers)
        self.device = torch.device(device)
        index = self.device.index if self.device.index is not None else torch.npu.current_device()
        self.index = index
        if not self.control.devctxs:
            if not self.control.init_device(index):
                raise RuntimeError('comfy-aimdo did not initialize device %d' % index)
        # aimdo initializes each device once per process. A second resolver in
        # the same process (another model, or the decoder) reuses that context;
        # ModelVBAR raises if this index was never initialized.

        # entries[position] = [(name, module_tensor, host_copy, dtype, shape, size), ...]
        self.entries, total = [], 0
        for layer in self.layers:
            rows = []
            for name, value in layer.named_parameters():
                if value.device.type != 'cpu' or value.requires_grad:
                    raise ValueError('Offloaded NPU parameters must be frozen CPU tensors')
                rows.append((name, value, value.detach(), value.dtype, tuple(value.shape),
                             value.numel() * value.element_size()))
                total += value.numel() * value.element_size()
            for name, value in layer.named_buffers():
                # A device-resident buffer is a precomputed modulation table: it
                # is not part of the streamed weights.
                if value.device.type != 'cpu' or value.requires_grad:
                    continue
                rows.append((name, value, value.detach(), value.dtype, tuple(value.shape),
                             value.numel() * value.element_size()))
                total += value.numel() * value.element_size()
            self.entries.append(rows)
        self.layer_bytes = [sum(row[5] for row in rows) for rows in self.entries]
        self.host_bytes = total
        self.vbar = ModelVBAR(total + int(slack_bytes), device=index)
        # alloc() aligns every slice, so the reservation is a little larger than
        # the sum of the tensor sizes; ask for slack proportional to slice count.
        self.allocations = [[self.vbar.alloc(row[5]) for row in rows][:]
                            for rows in self.entries]
        if watermark_bytes is not None:
            self.vbar.set_watermark_limit(int(watermark_bytes))
        self.signatures = [{} for _ in self.layers]
        self.resident = [{} for _ in self.layers]
        self.faulted = self.offloaded = self.populated = 0
        self.h2d_bytes = 0
        self.load_seconds = 0.
        self.active = None
        self.closed = False
        self.hooks = []
        if manage_hooks:
            for position, layer in enumerate(self.layers):
                self.hooks.append(layer.register_forward_pre_hook(
                    lambda module, args, position=position: self._before(position)))
                self.hooks.append(layer.register_forward_hook(
                    lambda module, args, result, position=position: self._after(position), always_call=True))

    def _before(self, position):
        started = time.monotonic()
        resident = {}
        for row, alloc in zip(self.entries[position], self.allocations[position]):
            name, module_tensor, host_copy, dtype, shape, size = row
            signature = self.fault(alloc)
            if signature is None:
                # The device is full: aimdo's documented fallback is a temporary
                # tensor that this one forward uses in place of the weight.
                resident[name] = host_copy.to(self.device)
                self.offloaded += 1
                self.h2d_bytes += size
                continue
            view = self.bridge.aimdo_to_tensor(alloc, self.device).view(dtype).view(shape)
            if not self.compare(signature, self.signatures[position].get(name)):
                view.copy_(host_copy)
                self.signatures[position][name] = signature
                self.populated += 1
                self.h2d_bytes += size
            resident[name] = view
            self.faulted += 1
        # Point the module at the device tensors only after every slice of this
        # layer is ready, so a partially faulted layer never executes.
        for row in self.entries[position]:
            row[1].data = resident[row[0]]
        self.resident[position] = resident
        self.active = position
        self.load_seconds += time.monotonic() - started

    def _after(self, position):
        for row, alloc in zip(self.entries[position], self.allocations[position]):
            name, module_tensor, host_copy, _dtype, _shape, _size = row
            held = self.resident[position].pop(name, None)
            if held is not None and held.data_ptr() == self._vbar_ptr(alloc):
                self.unpin(alloc)          # the VBAR slice may be evicted later
            module_tensor.data = host_copy  # a fallback tensor is simply dropped
        self.active = None

    def _vbar_ptr(self, alloc):
        return int(alloc[1])

    def stats(self):
        return {'device': str(self.device), 'residency': 'vbar', 'slots': 1, 'prefetch': False,
                'vbar_bytes': int(self.vbar.max_size), 'vbar_loaded_bytes': int(self.vbar.loaded_size()),
                'host_weight_bytes': self.host_bytes, 'layer_faults': self.faulted,
                'layer_offloads': self.offloaded, 'layer_populated': self.populated,
                'h2d_bytes': self.h2d_bytes, 'transfers': self.faulted + self.offloaded,
                'load_seconds': self.load_seconds, 'vbar_watermark': int(self.vbar.get_watermark()),
                'vbar_pages': int(self.vbar.get_nr_pages()),
                'cuda_buffer_bytes': 0, 'pass_cache_bytes': 0, 'pass_cache_hits': 0,
                'pass_cache_layer_count': 0, 'direct_read_layers': 0, 'direct_read_bytes': 0,
                'timing_note': 'Ascend faults and copies are synchronous and inside the sampled step.'}

    def cached_bytes(self):
        """Zero: the channel already holds what fits, page by page.

        An engine pass cache would be a second, smaller copy of the same idea, and
        the placement leaves 0.56 GiB spare against a 0.798 GiB block anyway. The
        diagnostics keep the key at zero rather than billing the channel's own
        resident bytes as an engine-owned cache.
        """
        return 0

    def cache_between_steps(self, *args, **kwargs):
        """Accepted for contract parity; ``cached_bytes`` explains why nothing is."""
        return 0

    def drive_layers(self):
        """Zero layers need re-reading: the channel faults from the host copies."""
        return 0

    def close(self):
        if self.closed:
            return
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        for rows in self.entries:
            for row in rows:
                row[1].data = row[2]
        self.resident = [[] for _ in self.entries]
        self.vbar = None
        self.active = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
