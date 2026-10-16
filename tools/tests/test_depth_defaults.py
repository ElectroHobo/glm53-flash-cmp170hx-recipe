"""Exercise the launcher's DFlash depth configuration through its DRY interface."""
import dataclasses
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(os.environ.get('RECIPE_TEST_ROOT', Path(__file__).resolve().parents[2]))
# The measured depth-3..7 step costs per layout (one request, several requests).
TABLES = {
    'tp4': ('1.0000,1.1009,1.1631,1.2214,1.2718', '1.0000,1.1132,1.1958,1.2214,1.2718'),
    'pp4': ('1.0000,1.0894,1.1375,1.2001,1.2239', '1.0000,1.0889,1.1194,1.2001,1.2239'),
}
# The verified depth for one request, two requests and more (the deepest is
# the number of draft slots), and the acceptance hysteresis.
BY_LOAD = [7, 5, 3]
HYSTERESIS = 0.03
DEPTH_CASES = [(base, one, two) for base, one, two in itertools.product(range(3, 8), repeat=3)
               if max(one, two) > base]
# The switches the speculative configuration replaced; a launch never sets them.
RETIRED = (
    'VLLM_GLM5_DFLASH_ADAPTIVE_K', 'VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS',
    'VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH', 'VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT',
    'VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS', 'VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI',
    'VLLM_GLM5_DFLASH_ADAPTIVE_K_HYST', 'VLLM_GLM5_DFLASH_ADAPTIVE_K_PRIOR',
    'VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG', 'VLLM_GLM5_LOCAL_LOGITS',
)


def costs(table, lo, hi):
    """The measured costs of depths lo..hi, indexed by absolute depth."""
    return [float(c) for c in table.split(',')[lo - 3:hi - 2]]


def expected_adaptive_k(layout):
    single, multi = TABLES[layout]
    value = {'by_load': BY_LOAD, 'accept': {'costs': costs(single, 3, 7),
                                            'costs_multi': costs(multi, 3, 7),
                                            'hysteresis': HYSTERESIS}, 'log_interval': 0}
    if layout == 'tp4':
        value['draft_by_load'] = True
    return value


def own_config(layout, base, one, two, accept=True, width=None):
    """A caller's adaptive mapping for depths one, two and base under load."""
    lo, hi = min(base, one, two), max(base, one, two)
    single, multi = TABLES[layout]
    adaptive = {'by_load': [one, two, base]}
    if accept:
        adaptive['accept'] = {'costs': costs(single, lo, hi), 'costs_multi': costs(multi, lo, hi)}
    if width is not None:
        adaptive['draft_by_load'] = width
    return {'method': 'dflash', 'model': '/own/drafter', 'num_speculative_tokens': hi,
            'adaptive_k': adaptive}


@pytest.fixture(autouse=True)
def cpu_launcher(tmp_path, monkeypatch):
    package = tmp_path / 'vllm'
    package.mkdir()
    (package / '__init__.py').touch()
    (package / '_ampere_marlin_C.py').touch()
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin/python').symlink_to(sys.executable)
    monkeypatch.setenv('VENV', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    monkeypatch.setenv('RECIPE_CACHE_ROOT', str(tmp_path / 'cache'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('P2P', 'force')


def run(layout, root=ROOT, mode=None, **overrides):
    env = {k: v for k, v in os.environ.items()
           if k not in RETIRED and k not in
           ('PP', 'TP', 'LAYOUT', 'SPEC_N', 'EXTRA_ARGS', 'MAX_BATCHED')}
    env.update(DRY='1', LAYOUT=layout, VLLM_AMPERE_MARLIN_DECODE_CUDA='0',
               MODEL='/unused/target', DFLASH_MODEL='/unused/drafter')
    env.update(overrides)
    return subprocess.run(['bash', str(root / 'serve.sh')] + ([mode] if mode else []), env=env,
                          capture_output=True, text=True, timeout=30)


def launch(layout, root=ROOT, mode=None, **overrides):
    result = run(layout, root, mode, **overrides)
    assert result.returncode == 0, result.stderr
    dumped = dict(line.strip().split('=', 1) for line in result.stdout.splitlines()
                  if line.strip().startswith(('VLLM_', 'GLM5_', 'NCCL_', 'PYTORCH_',
                                              'PYTHONPATH=', 'FLASHINFER_', 'CUDA_HOME=')))
    command = next(line for line in result.stdout.splitlines()
                   if ' --served-model-name ' in line)
    args = shlex.split(command)
    specs = [json.loads(args[i + 1]) for i, arg in enumerate(args) if arg == '--speculative-config']
    assert len(specs) <= 1, 'at most one --speculative-config'
    raw = specs[0] if specs else None
    pp = int(args[args.index('--pipeline-parallel-size') + 1])
    return dumped, raw, pp, result.stdout, args


def launch_own(layout, config, **overrides):
    return launch(layout, EXTRA_ARGS='--speculative-config ' + json.dumps(config, separators=(',', ':')),
                  **overrides)


@pytest.mark.parametrize('layout', TABLES)
def test_ordinary_defaults_unchanged(layout):
    current = launch(layout)
    values, raw, pp, stdout, args = current
    assert raw == {'method': 'dflash', 'model': '/unused/drafter', 'num_speculative_tokens': 7,
                   'adaptive_k': expected_adaptive_k(layout)}
    assert pp == (4 if layout == 'pp4' else 1)
    assert args[args.index('--max-num-batched-tokens') + 1] == ('3463' if layout == 'tp4' else '2312')
    assert 'draft slots=7' in stdout
    assert not set(RETIRED).intersection(values)
    reference = os.environ.get('RECIPE_REFERENCE_ROOT')
    if reference:
        previous = launch(layout, root=Path(reference))
        assert current[1:3] == previous[1:3]


@pytest.mark.parametrize('layout', TABLES)
def test_default_costs_are_indexed_by_absolute_depth(layout):
    _, raw, _, _, _ = launch(layout)
    accept = raw['adaptive_k']['accept']
    lo, hi = min(BY_LOAD), max(BY_LOAD)
    assert len(accept['costs']) == len(accept['costs_multi']) == hi - lo + 1
    for key, table in zip(('costs', 'costs_multi'), TABLES[layout]):
        assert accept[key] == costs(table, lo, hi)


@pytest.mark.parametrize('layout', TABLES)
def test_batch_sharded_sampling_is_a_tp_option(layout):
    _, _, _, _, args = launch(layout)
    assert ('--enable-batch-sharded-sampling' in args) == (layout == 'tp4')


@pytest.mark.parametrize('layout', TABLES)
def test_spec_n_is_refused_for_the_adaptive_mapping(layout):
    result = run(layout, SPEC_N='3')
    assert result.returncode == 2
    assert 'SPEC_N applies to mtp only' in result.stderr


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('base,one,two', DEPTH_CASES)
def test_documented_depth_range_dry_run(layout, base, one, two):
    config = own_config(layout, base, one, two)
    values, raw, _, stdout, _ = launch_own(layout, config)
    assert raw == config
    assert 'adaptive_k=from EXTRA_ARGS' in stdout
    assert not set(RETIRED).intersection(values)


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('explicit', [[], [1, 2, 3], [1, 0, 2], 'not,a,table'])
def test_explicit_values_remain_owned_by_the_user(layout, explicit):
    config = own_config(layout, 3, 5, 5)
    config['adaptive_k']['accept']['costs'] = explicit
    _, raw, _, _, _ = launch_own(layout, config)
    assert raw == config


@pytest.mark.parametrize('layout', TABLES)
def test_fixed_depth_kill_switch(layout):
    own = {'method': 'dflash', 'model': '/own/drafter', 'num_speculative_tokens': 3}
    values, raw, _, stdout, args = launch_own(layout, own, SPEC_N='3')
    assert raw == own
    assert 'adaptive_k=from EXTRA_ARGS draft slots=3' in stdout
    assert args[args.index('--max-num-batched-tokens') + 1] == ('3460' if layout == 'tp4' else '2312')
    assert not set(RETIRED).intersection(values)


@pytest.mark.parametrize('layout', TABLES)
def test_mtp_keeps_spec_n(layout):
    _, raw, _, stdout, _ = launch(layout, mode='mtp', SPEC_N='4')
    assert raw == {'method': 'mtp', 'num_speculative_tokens': 4}
    assert 'adaptive_k=off draft slots=4' in stdout


@pytest.fixture(scope='module')
def engine():
    """Native-runtime checks use the pinned fork, without loading a model."""
    source = os.environ.get('RECIPE_TEST_ENGINE')
    if source:
        sys.path.insert(0, source)
    module = pytest.importorskip('vllm.config.speculative',
                                 reason='install the pinned native fork for CPU engine-contract tests')
    import torch
    import vllm.config.vllm as vllm_config
    import vllm.sm80.envs as sm80_envs
    from vllm.sm80.spec_decode.adaptive_k import AdaptiveKConfig
    assert not torch.cuda.is_initialized()
    return SimpleNamespace(speculative=module, vllm_config=vllm_config, envs=sm80_envs,
                           policy=AdaptiveKConfig, torch=torch)


def consume(engine, raw):
    """Run the speculative-config validation that parses adaptive_k."""
    cls = engine.speculative.SpeculativeConfig
    fields = {f.name: f.default for f in dataclasses.fields(cls)
              if f.default is not dataclasses.MISSING and not hasattr(f.default, 'json_schema_extra')}
    config = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(config, name, value)
    for name in ('tensor_parallel_size', 'draft_model_config', 'adaptive_k_config',
                 'target_model_config', 'num_speculative_tokens_per_batch_size',
                 'synthetic_acceptance_rates', 'synthetic_acceptance_length'):
        object.__setattr__(config, name, None)
    object.__setattr__(config, 'use_heterogeneous_vocab', False)
    object.__setattr__(config, 'enable_adaptive_verification',
                       raw.get('enable_adaptive_verification', False))
    object.__setattr__(config, 'method', raw['method'])
    object.__setattr__(config, 'num_speculative_tokens', raw['num_speculative_tokens'])
    object.__setattr__(config, 'adaptive_k', raw.get('adaptive_k'))
    assert cls._verify_args(config) is config
    assert not engine.torch.cuda.is_initialized()
    return config


def test_retired_names_are_the_engines(engine):
    for name in RETIRED:
        assert name in engine.envs.RETIRED_ENV_VARS, name


@pytest.mark.parametrize('layout', TABLES)
def test_engine_accepts_the_layout_mapping(engine, layout):
    values, raw, _, _, _ = launch(layout)
    config = consume(engine, raw)
    policy = config.adaptive_k_config
    single, multi = TABLES[layout]
    assert isinstance(policy, engine.policy)
    assert config.uses_adaptive_k()
    assert policy.by_load == tuple(BY_LOAD)
    assert policy.allowed == config.adaptive_k_draft_counts() == (3, 4, 5, 6, 7)
    assert [policy.k_for_load(n) for n in (1, 2, 3, 8)] == [7, 5, 3, 3]
    assert policy.draft_by_load == (layout == 'tp4')
    assert policy.accept
    assert policy.accept_costs == tuple(costs(single, 3, 7))
    assert policy.accept_costs_multi == tuple(costs(multi, 3, 7))
    assert policy.accept_hysteresis == HYSTERESIS
    assert policy.log_interval == 0
    assert not set(engine.envs.RETIRED_ENV_VARS).intersection(values)


@pytest.mark.parametrize('layout', TABLES)
def test_engine_keeps_adaptive_depth_without_data_parallelism(engine, layout):
    _, raw, _, _, _ = launch(layout)
    config = consume(engine, raw)
    host = SimpleNamespace(speculative_config=config,
                           parallel_config=SimpleNamespace(data_parallel_size=1))
    engine.vllm_config.VllmConfig._maybe_disable_dynamic_sd_for_data_parallel(host)
    assert config.uses_adaptive_k()
    assert config.adaptive_k == raw['adaptive_k']


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('base,one,two', DEPTH_CASES)
def test_documented_depth_range_in_engine(engine, layout, base, one, two):
    config = own_config(layout, base, one, two)
    _, raw, _, _, _ = launch_own(layout, config)
    policy = consume(engine, raw).adaptive_k_config
    lo, hi = min(base, one, two), max(base, one, two)
    assert policy.accept
    assert policy.allowed == tuple(range(lo, hi + 1))
    assert policy.accept_costs == tuple(config['adaptive_k']['accept']['costs'])
    assert policy.accept_costs_multi == tuple(config['adaptive_k']['accept']['costs_multi'])
    assert [policy.k_for_load(n) for n in (1, 2, 3)] == [one, two, base]


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('accept,width', list(itertools.product([False, True], repeat=2)))
def test_documented_accept_and_width_switches(engine, layout, accept, width):
    config = own_config(layout, 3, 5, 5, accept=accept, width=width)
    _, raw, _, _, _ = launch_own(layout, config)
    policy = consume(engine, raw).adaptive_k_config
    assert policy.accept == accept
    assert policy.draft_by_load == width
    assert policy.allowed == ((3, 4, 5) if accept else (3, 5))


@pytest.mark.parametrize('layout', TABLES)
@pytest.mark.parametrize('base', range(3, 8))
def test_fixed_depth_off_path_is_inert(engine, layout, base):
    own = {'method': 'dflash', 'model': '/own/drafter', 'num_speculative_tokens': base}
    values, raw, _, _, _ = launch_own(layout, own, SPEC_N=str(base))
    config = consume(engine, raw)
    assert config.adaptive_k is None
    assert not config.uses_adaptive_k()
    assert config.adaptive_k_draft_counts() == ()
    assert not set(engine.envs.RETIRED_ENV_VARS).intersection(values)
