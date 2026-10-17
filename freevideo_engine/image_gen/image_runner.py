"""One process's loaded image model and the per-request placement.

``QwenImageGenerator`` (qwen_pipeline) knows the model; this module decides
where the parts live. The DiT blocks and the text-encoder layers are frozen
CPU weights streamed through ``StreamedStack`` (the engine's LayerOffloader);
what stays on the card between uses comes from ``plan_image``. The vocabulary
table is looked up on the CPU, the VAE and the vision tower visit the card
only for the stage that needs them.
"""
import time
from pathlib import Path

import torch

from ..torch_device import device_module
from . import image_policy
from .image_stream import StreamedStack

GiB = 2**30
_DIT_FILE = 'diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors'
_TE_FILE = 'text_encoders/qwen3vl_8b_int8_convrot.safetensors'
_VAE_FILE = 'vae/qwen_image_2.1_vae_bf16.safetensors'


def _non_block_children(module, skip):
    for name, child in module.named_children():
        if name != skip:
            yield child


class _Placement:
    """Stage hooks for QwenImageGenerator: nothing but moves between host and card."""

    def __init__(self, runner):
        self.runner = runner

    def enter(self, stage):
        runner = self.runner
        if stage == 'text':
            visual = getattr(runner.text_model.model, 'visual', None)
            if visual is not None:
                visual.to(runner.device)
        elif stage in ('encode', 'decode'):
            runner.vae.to(runner.device)

    def leave(self, stage):
        runner = self.runner
        if stage == 'text':
            visual = getattr(runner.text_model.model, 'visual', None)
            if visual is not None:
                visual.to('cpu')
            self._trim()
        elif stage in ('encode', 'decode'):
            runner.vae.to('cpu')
            self._trim()
        elif stage == 'denoise':
            self._trim()

    @staticmethod
    def _trim():
        # Stage peaks are planned on allocated bytes; give the cached blocks back so the
        # next stage's workspace is not fragmented by the previous one's.
        if device_module(torch).is_available():
            device_module(torch).empty_cache()


class ImageRunner:
    """Load the pack once, then ``generate`` repeatedly.

    ``gpu_budget_bytes`` caps what a request may plan against (None reads the
    card: free memory plus what this process already holds). Residency and VAE
    tiling are chosen per request by ``image_policy.plan_image``.
    """

    def __init__(self, pack_dir, *, device='cuda', gpu_budget_bytes=None, log=None, progress=None):
        from .qwen_pipeline import QwenImageGenerator
        from .qwen_te_weights import load_text_encoder
        from .qwen_weights import load_dit, load_vae

        self.pack_dir = Path(pack_dir)
        self.device = torch.device(device)
        self.gpu_budget_bytes = gpu_budget_bytes
        self.log = log or (lambda *args, **kwargs: None)
        self.load_seconds = {}

        def path(directory, name):
            nested = self.pack_dir / directory / name
            return nested if nested.exists() else self.pack_dir / name

        progress = progress or (lambda stage, fraction: None)
        progress('load', 0.)
        started = time.monotonic()
        self.transformer = load_dit(self.pack_dir / _DIT_FILE, device='cpu')
        self.load_seconds['dit'] = time.monotonic() - started
        progress('load', 1 / 3)
        started = time.monotonic()
        self.vae = load_vae(self.pack_dir / _VAE_FILE, device='cpu')
        self.load_seconds['vae'] = time.monotonic() - started
        progress('load', 2 / 3)
        started = time.monotonic()
        config_dir = self.pack_dir / 'text_encoder'
        processor_dir = self.pack_dir / 'processor'
        text = load_text_encoder(self.pack_dir / _TE_FILE, config_path=config_dir / 'config.json', device='cpu')
        from transformers import Qwen3VLProcessor
        processor = Qwen3VLProcessor.from_pretrained(str(processor_dir), local_files_only=True)
        self.load_seconds['text'] = time.monotonic() - started
        self.text_model = text

        # Parts that always sit on the card: everything of the DiT except its blocks, and the
        # encoder's final norm and rotary table. The 1.2 GB vocabulary stays on the host and is
        # looked up there (a token id list is tiny), which keeps it out of the encode peak.
        for child in _non_block_children(self.transformer, 'transformer_blocks'):
            child.to(self.device)
        for parameter in self.transformer.parameters(recurse=False):
            parameter.data = parameter.data.to(self.device)
        language = text.model.language_model
        for name, child in language.named_children():
            if name not in ('layers', 'embed_tokens'):
                child.to(self.device)
        self._embed_hooks = self._lookup_on_host(language.embed_tokens)

        started = time.monotonic()
        with torch.no_grad():   # LayerOffloader refuses to be built with autograd on
            self.dit_stack = StreamedStack(self.transformer.transformer_blocks, name='dit', device=self.device)
            self.te_stack = StreamedStack(language.layers, name='text_encoder', device=self.device)
        self.load_seconds['pin'] = time.monotonic() - started

        self.shape = image_policy.ImageShape.measure(
            list(self.transformer.transformer_blocks), list(language.layers),
            total_weight_bytes=self.dit_stack.pinned_bytes + self.te_stack.pinned_bytes)
        self.generator = QwenImageGenerator(
            None, placement=_Placement(self), device=self.device, transformer=self.transformer, vae=self.vae,
            text_encoder=text, processor=processor, sample_sigmas=None, vae_tile=True)
        from .qwen_pipeline import load_sample_sigmas
        self.generator.sample_sigmas = load_sample_sigmas(self.pack_dir / _DIT_FILE)
        self.last_plan = None
        progress('load', 1.)

    def _lookup_on_host(self, embedding):
        device = self.device

        def to_host(module, args):
            return (args[0].to('cpu'),) + tuple(args[1:])

        def to_card(module, args, output):
            return output.to(device)

        return [embedding.register_forward_pre_hook(to_host), embedding.register_forward_hook(to_card)]

    def budget(self):
        if self.gpu_budget_bytes is not None:
            return int(self.gpu_budget_bytes)
        free, _ = device_module(torch).mem_get_info(self.device)
        return int(free + device_module(torch).memory_reserved(self.device))

    def plan(self, width, height, reference_images=None):
        width, height = width // 32 * 32, height // 32 * 32
        reference_pixels = 0
        resolution = self.generator.reference_resolution
        for image in reference_images or []:
            scale = (resolution * resolution / (image.width * image.height)) ** 0.5
            reference_pixels += round(image.width * scale) * round(image.height * scale)
        try:
            from ..offload import live_pin_capacity
            host_available = live_pin_capacity() + self.dit_stack.pinned_bytes + self.te_stack.pinned_bytes
        except Exception:
            host_available = None
        return image_policy.plan_image(
            gpu_budget_bytes=self.budget(), host_available_bytes=host_available, width=width, height=height,
            reference_pixels=reference_pixels, shape=self.shape)

    def generate(self, prompt, *, width, height, seed=None, reference_images=None, progress=None, cancel=None):
        plan = self.plan(width, height, reference_images)
        self.last_plan = plan
        if not plan.feasible:
            raise MemoryError('; '.join(plan.reasons) or 'the image model does not fit this machine')
        self.dit_stack.set_resident(plan.dit_resident_layers)
        self.te_stack.set_resident(plan.te_resident_layers)
        self.generator.vae_tile = plan.vae_tiling
        return self.generator.generate(prompt, width=width, height=height, seed=seed,
                                       reference_images=reference_images, progress=progress, cancel=cancel)

    def stats(self):
        return dict(dit=self.dit_stack.stats(), text_encoder=self.te_stack.stats(),
                    plan=self.last_plan.to_dict() if self.last_plan else None, load_seconds=self.load_seconds)

    def close(self):
        for hook in self._embed_hooks:
            hook.remove()
        self.generator.close()
        self.dit_stack.close()
        self.te_stack.close()
        self.transformer = self.vae = self.text_model = None
