"""Synchronous Ascend layer residency for models larger than device memory.

The CUDA offloader overlaps host-to-device copies with compute using streams,
events and pinned staging buffers. Torch-NPU exposes none of those through this
engine yet, so a layer is copied into one shared per-dtype slot immediately
before its forward and released to its host copy after it. That trades the
overlap for a bound: at most one layer's weights are on the device at a time,
and its size is reported instead of assumed.

Precomputed modulation tables that already live on the device are left in place,
exactly as the CUDA offloader leaves them, so AdaLN caching stays valid.
"""
import time

import torch


class NPULayerResidency:
    """Host-resident weights with one reusable device slot per dtype.

    The layers keep their own frozen CPU tensors between forwards; the hooks only
    re-point each tensor's storage at a view of the shared slot. Nothing about
    the module graph changes, so a later step sees the same host copies.
    """

    def __init__(self, layers, device='npu', prefetch=True, manage_hooks=True, weight_source=None):
        if torch.is_grad_enabled():
            raise RuntimeError('NPU layer residency requires gradients disabled')
        if weight_source is not None:
            raise NotImplementedError('NPU residency keeps host copies; streamed weight sources are not integrated')
        self.layers = list(layers)
        self.device = torch.device(device)
        # Accepted for contract parity with the CUDA offloader. Ascend copies are
        # synchronous, so a second slot would only reserve bytes.
        self.prefetch = bool(prefetch)
        self.weight_source = None
        self.cpu, self.layout, self.layer_bytes = [], [], []
        sizes = {}
        for layer in self.layers:
            weights, entries, offsets, total = {}, [], {}, 0
            for name, value in layer.named_parameters():
                if value.device.type != 'cpu' or value.requires_grad:
                    raise ValueError('Offloaded NPU parameters must be frozen CPU tensors')
                entries.append(self._entry(value, name, offsets))
                weights[name] = value.detach()
            for name, value in layer.named_buffers():
                # A device-resident buffer is a precomputed modulation table; it
                # is not part of the streamed weights.
                if value.device.type != 'cpu' or value.requires_grad:
                    continue
                entries.append(self._entry(value, name, offsets))
                weights[name] = value
            for dtype, count in offsets.items():
                sizes[dtype] = max(sizes.get(dtype, 0), count)
            for value, *_rest in entries:
                total += value.numel() * value.element_size()
            self.cpu.append(weights)
            self.layout.append(entries)
            self.layer_bytes.append(total)
            del weights, entries
        self.host_bytes = sum(self.layer_bytes)
        self.slot = {dtype: torch.empty(count, dtype=dtype, device=self.device)
                     for dtype, count in sizes.items()}
        self.slot_bytes = sum(value.numel() * value.element_size() for value in self.slot.values())
        self.loads = 0
        self.h2d_bytes = 0
        self.load_seconds = 0.
        self.active = None
        self.closed = False
        self.hooks = []
        if manage_hooks:
            for index, layer in enumerate(self.layers):
                self.hooks.append(layer.register_forward_pre_hook(
                    lambda module, args, index=index: self._before(index)))
                self.hooks.append(layer.register_forward_hook(
                    lambda module, args, result, index=index: self._after(index), always_call=True))

    @staticmethod
    def _entry(value, name, offsets):
        dtype = value.dtype
        offset = offsets.get(dtype, 0)
        offsets[dtype] = offset + value.numel()
        return (value, name, dtype, offset, value.numel(), tuple(value.shape))

    def _before(self, index):
        started = time.monotonic()
        for value, name, dtype, offset, count, shape in self.layout[index]:
            view = self.slot[dtype].narrow(0, offset, count).view(shape)
            view.copy_(self.cpu[index][name])
            value.data = view
        self.active = index
        self.loads += 1
        self.h2d_bytes += self.layer_bytes[index]
        self.load_seconds += time.monotonic() - started

    def _after(self, index):
        for value, name, *_rest in self.layout[index]:
            value.data = self.cpu[index][name]
        self.active = None

    def cached_bytes(self):
        """Zero, and not as a stand-in: no pass cache could fit.

        The placement keeps as many blocks device-resident as the budget allows
        and faults the rest, which on a 61.3 GiB device leaves 0.56 GiB spare
        against a 0.798 GiB block. There is no room a cross-step cache could take,
        so reporting zero is the measurement rather than a missing feature.
        """
        return 0

    def cache_between_steps(self, *args, **kwargs):
        """Accepted for contract parity; ``cached_bytes`` explains why nothing is."""
        return 0

    def drive_layers(self):
        """Zero layers need re-reading: every offloaded block keeps a host copy."""
        return 0

    def stats(self):
        return {'device': str(self.device), 'residency': 'host-copies', 'prefetch': self.prefetch, 'slots': 1,
                'streamed_checkpoint_layers': False, 'streamed_layer_count': 0,
                'pinned_model_sources': False, 'pinned_layer_count': 0, 'pinned_layer_indices': [],
                'slot_bytes': self.slot_bytes, 'host_weight_bytes': self.host_bytes,
                'layer_loads': self.loads, 'h2d_bytes': self.h2d_bytes,
                'transfers': self.loads, 'load_seconds': self.load_seconds,
                # Keys the shared diagnostics read; the CUDA-only budgets stay 0.
                'cuda_buffer_bytes': 0, 'pass_cache_bytes': 0, 'pass_cache_hits': 0,
                'pass_cache_layer_count': 0, 'direct_read_layers': 0, 'direct_read_bytes': 0,
                'timing_note': 'Ascend copies are synchronous and inside the sampled step; '
                               'load_seconds is not overlapped with compute.'}

    def close(self):
        if self.closed:
            return
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        for index, entries in enumerate(self.layout):
            for value, name, *_rest in entries:
                value.data = self.cpu[index][name]
        self.slot.clear()
        self.active = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
