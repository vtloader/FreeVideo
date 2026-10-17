"""Compile a request shape's sampling kernels before its first generation.

The first sampling step of a new engine compiles the fused block kernels and
tunes the int8 projections for the request's plan: the head group, the frame
layout and the projection tiles all reach the compiled shapes. All of that
happens in the first block of the first step, and it is the same in every
later block and step. On a 12 GB Windows laptop it took about 100 s of the
first request after an update; afterwards the Triton and inductor caches in
kernel-cache serve every process.

This worker loads the engine with a planned request's placement, runs each
sampling pass until the second block is entered, and stops. Then, with the
engine closed as before a decode, it runs the latent upscale of a two-pass
plan and decodes one temporal clip at the request's canvas: the VAE decodes
clips of seven latent frames in spatial tiles laid out from the canvas, so a
22-frame video reaches every tile shape of a long one. On Windows the GPU
driver also finalizes each loaded kernel module once per driver version, which
these steps cover for the modules they load.

The prompt is zeros of the encoder's width: row counts are compiled
dynamically, and the values never reach a kernel's shape. No video is kept,
and no resource history, compatibility or input cache is written.
"""
from .torch_device import device_module, device_name
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import torch

from .locking import runtime_lock
from .monitoring import save
from .processes import worker_signals


class FirstBlockDone(Exception):
    pass


@contextmanager
def first_block(engine):
    """Stop sampling when the second transformer block is entered."""
    def stop(module, arguments):
        if device_module(torch).is_initialized():
            device_module(torch).synchronize()
        raise FirstBlockDone()
    handle = engine.transformer.transformer_blocks[1].register_forward_pre_hook(stop)
    try:
        yield
    finally:
        handle.remove()


# The shortest video whose latents (seven frames) decode as one whole VAE clip.
CLIP_FRAMES = 22


def zero_latents(channels, canvas):
    """Zero video and audio latents in the shapes sampling produces for a canvas."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_CHANNELS, align_num_frames, audio_latent_num_frames, video_latent_num_frames)
    frames = align_num_frames(canvas['frames'], 17, 5)
    device = device_name(torch)
    video = torch.zeros((1, channels, video_latent_num_frames(frames, 17, 5),
                         canvas['height'] // 16, canvas['width'] // 16), device=device)
    audio = torch.zeros((MINIMAX_H3_AUDIO_CHANNELS, 32, audio_latent_num_frames(frames)), device=device)
    return video, audio


def upscale_once(plan, channels):
    """A two-pass plan's latent upscale; None when setup has no upscaler checkpoint."""
    from .latent_upscale import upscale
    from .two_pass import checkpoint_path
    path = checkpoint_path()
    if not path.is_file():
        return None
    tick = time.perf_counter()
    video, _ = zero_latents(channels, plan['first'])
    upscale(video, str(path), plan['upscale_target']['width'], plan['upscale_target']['height'])
    del video
    device_module(torch).empty_cache()
    return time.perf_counter() - tick


def decode_clip(base, decoder, channels, canvas, path):
    """Decode one clip at the request's canvas with the request's decoder placement."""
    from .decode import decode_to_file
    tick = time.perf_counter()
    video, audio = zero_latents(channels, dict(canvas, frames=CLIP_FRAMES))
    decode_to_file(video, audio, str(path), base=base, **decoder)
    del video, audio
    device_module(torch).empty_cache()
    return time.perf_counter() - tick


def first_pass(engine, conditioning, request):
    """Run one pass until its second block; seconds taken."""
    tick = time.perf_counter()
    try:
        with first_block(engine):
            engine.sample(conditioning, 0, **request)
    except FirstBlockDone:
        if device_module(torch).is_initialized():
            device_module(torch).empty_cache()
        return time.perf_counter() - tick
    raise RuntimeError('Sampling finished without entering a second block')


@torch.no_grad()
def compile_request(request):
    from .runtime import Engine
    from .worker import device_memory_report, prefetch_compiler_keys
    from .precompile import worker_options
    torch.set_num_threads(8)
    prefetch_compiler_keys()
    receipt = dict(success=False, geometry=request['geometry'], passes=[])
    save(request['receipt'], receipt)
    receipt['device_memory'] = device_memory_report(request['gpu_budget_bytes'], request.get('allocator_limit_bytes'),
                                                    reserve_bytes=request.get('gpu_reserve_bytes', 0),
                                                    device=request.get('device', 'cuda'))
    allocation = receipt['device_memory'].get('effective_allocator_limit_bytes') or request['gpu_budget_bytes']
    tick = time.perf_counter()
    # A missing sampling table becomes zeros of its shape: it is installed by
    # the request that uses it, never by setup.
    engine = Engine(request['cache'], gpu_allocation_bytes=int(allocation), fetch_tables=False,
                    **dict(worker_options(request['engine_options']), canvas=request['geometry']))
    receipt['load_seconds'] = time.perf_counter() - tick
    plan = request['sampling_plan']
    channels, base = engine.transformer.config.in_channels, engine.base
    final = plan['second'] if plan.get('enabled') else plan['first']
    try:
        reserve = dict(gpu_reserve_bytes=request.get('gpu_reserve_bytes', 0))
        options = (request.get('first_pass_policy') or {}).get('profile', {}).get('engine')
        # One prompt length per recent request: the int8 projections tune one
        # kernel per power-of-two row band, and the prompt moves the bands of
        # the tiles it falls into. Later lengths reuse every compiled graph.
        for conditioning in request['conditioning']:
            if plan.get('enabled'):
                seconds = first_pass(engine, conditioning, dict(plan['first'], allow_smaller_canvas=True,
                    compute_options=options, pass_resident_blocks=options.get('resident_blocks') if options else None,
                    **reserve))
                receipt['passes'].append(dict(canvas=plan['first'], conditioning=conditioning, seconds=seconds))
                seconds = first_pass(engine, conditioning, dict(final, initial_latents=zero_latents(channels, final),
                    refine_steps=plan['refine_steps'], refine_schedule=plan.get('refine_schedule'), **reserve))
            else:
                seconds = first_pass(engine, conditioning, dict(final, **reserve))
            receipt['passes'].append(dict(canvas=final, conditioning=conditioning, seconds=seconds))
    finally:
        # Tables compiled with zeros in their shape; their requests install them.
        receipt['table_placeholders'] = len(engine.table_placeholders)
        receipt['table_downloaded_bytes'] = engine.config['adaln_downloaded_bytes']
        engine.close()
        save(request['receipt'], receipt)
    if plan.get('enabled') and plan.get('upscaler_sha256'):
        receipt['upscale_seconds'] = upscale_once(plan, channels)
    if request.get('decoder_options') is not None:
        receipt['decode_seconds'] = decode_clip(base, request['decoder_options'], channels, final,
                                                Path(request['receipt']).with_suffix('.mp4'))
    receipt['success'] = True
    save(request['receipt'], receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    from .precompile import BACKGROUND_ENV, lower_priority
    if os.environ.get(BACKGROUND_ENV):
        lower_priority()
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    with worker_signals(), runtime_lock():
        compile_request(request)


if __name__ == '__main__':
    main()
