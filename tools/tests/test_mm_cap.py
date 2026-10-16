"""MM_CAP=1 must bound real images to what the engine reserved for them.

The GLM-5.3 image and video processors size every item from
``max_image_tokens`` and ignore ``max_pixels``; the engine's memory profiler
reads ``max_image_tokens`` too. The launcher therefore passes the token bound,
or real images larger than the encoder cache are refused with HTTP 400.

The launcher tests run everywhere. The processor test needs the engine and the
target checkpoint: set ``GLM53_MODEL_DIR`` to the checkpoint directory.
"""
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(os.environ.get('RECIPE_TEST_ROOT', Path(__file__).resolve().parents[2]))
MODEL_DIR = os.environ.get('GLM53_MODEL_DIR')
# One vision token covers a 28x28 patch-merge cell of the image.
PIXELS_PER_TOKEN = 14 * 14 * 2 * 2


@pytest.fixture
def launcher_env(tmp_path):
    """A CPU-only environment in which serve.sh DRY=1 runs without the engine."""
    package = tmp_path / 'vllm'
    package.mkdir()
    (package / '__init__.py').touch()
    (package / '_ampere_marlin_C.py').touch()
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin/python').symlink_to(sys.executable)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('MM_') and k != 'EXTRA_ARGS'}
    env.update(VENV=str(tmp_path), PYTHONPATH=str(tmp_path),
               RECIPE_CACHE_ROOT=str(tmp_path / 'cache'), CUDA_VISIBLE_DEVICES='',
               P2P='force', DRY='1', LAYOUT='tp4', VLLM_AMPERE_MARLIN_DECODE_CUDA='0',
               MODEL='/unused/target', DFLASH_MODEL='/unused/drafter')
    return env


def mm_flags(env, **overrides):
    result = subprocess.run(['bash', str(ROOT / 'serve.sh')], env={**env, **overrides},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-2000:]
    command = next(line for line in result.stdout.splitlines()
                   if ' --served-model-name ' in line)
    args = shlex.split(command)
    return {arg: json.loads(args[index + 1]) for index, arg in enumerate(args)
            if arg in ('--mm-processor-kwargs', '--limit-mm-per-prompt')}


def test_default_passes_no_multimodal_bound(launcher_env):
    assert mm_flags(launcher_env) == {}


@pytest.mark.parametrize('overrides,tokens', [
    ({}, 1280),
    ({'MM_MAX_PIXELS': str(2560 * 1440)}, 2560 * 1440 // PIXELS_PER_TOKEN),
    ({'MM_MAX_IMAGE_TOKENS': '4000'}, 4000),
    ({'MM_MAX_IMAGE_TOKENS': '8000'}, 8000),
])
def test_cap_passes_only_the_token_bound(launcher_env, overrides, tokens):
    kwargs = mm_flags(launcher_env, MM_CAP='1', **overrides)['--mm-processor-kwargs']
    assert kwargs == {'max_image_tokens': tokens}


@pytest.mark.skipif(not MODEL_DIR, reason='set GLM53_MODEL_DIR to the target checkpoint')
@pytest.mark.parametrize('tokens', ['1280', '4000', '8000'])
@pytest.mark.parametrize('size', [(3840, 2160), (2560, 1440), (1920, 1080)])
def test_real_image_fits_the_profiled_item(launcher_env, tokens, size):
    from PIL import Image
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

    flags = mm_flags(launcher_env, MM_CAP='1', MM_MAX_IMAGE_TOKENS=tokens)
    kwargs = flags['--mm-processor-kwargs']
    config = ModelConfig(model=MODEL_DIR, mm_processor_kwargs=kwargs,
                         limit_mm_per_prompt=flags['--limit-mm-per-prompt'],
                         max_model_len=65536)
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    profiled = get_mm_max_toks_per_item(processor, {'image': 1, 'video': 1}, None)
    hf = processor.info.get_hf_processor()
    out = processor.info.ctx.call_hf_processor(
        hf, {'text': hf.image_token, 'images': [Image.new('RGB', size, (200, 100, 50))]},
        dict(kwargs))
    real = int(out['image_grid_thw'].prod()) // 4
    assert real <= int(tokens)
    assert real <= profiled['image'], (real, dict(profiled))


# Engines whose profiler sizes MM_CAP=1 items from max_pixels only. With this launcher their
# profile falls back to the uncapped size, so MM_CAP=1 would save no memory: the token-only
# bound ships only with a later engine pin, never on these.
PROFILER_IGNORES_MAX_IMAGE_TOKENS = {'ab60b723ada254a442a4ba5ff27bf837aa27ef83'}


def test_pinned_engine_profiler_reads_the_token_bound():
    pin = re.search(r'^RELEASE_VLLM_COMMIT=(.+)$', (ROOT / 'start.sh').read_text(), re.M)[1]
    assert pin.strip('"\' ') not in PROFILER_IGNORES_MAX_IMAGE_TOKENS, (
        'MM_CAP=1 passes only max_image_tokens; pin an engine whose profiler reads it')
