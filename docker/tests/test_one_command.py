"""One command from a fresh clone: Compose, checkpoint completeness, native migration."""
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / 'docker' / 'prepare.sh'


def release_image():
    return re.search(r'^RELEASE_IMAGE=(.+)$', (ROOT / 'start.sh').read_text(), re.M)[1]


def test_every_default_image_is_the_start_pin():
    image = release_image()
    compose = (ROOT / 'docker-compose.yml').read_text()
    assert re.findall(r'\$\{IMAGE:-([^}]+)\}', compose) == [image]
    # Release history may name earlier images; nothing else may default to one.
    history = {ROOT / 'CHANGELOG.md', ROOT / 'docker' / 'RELEASE.md'}
    files = subprocess.run(['git', '-C', str(ROOT), 'ls-files'], capture_output=True,
                           text=True, check=True).stdout.split()
    for path in (ROOT / name for name in files):
        if path in history or 'tests' in path.parts or not path.is_file():
            continue
        for found in re.findall(r'ghcr\.io/morrowmake/vllm-cmp170hx@sha256:[0-9a-f]{64}',
                                path.read_text(errors='replace')):
            assert found == image, path


def test_compose_runs_the_start_container_after_prepare():
    yaml = pytest.importorskip('yaml')
    spec = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    prepare, server = spec['services']['prepare'], spec['services']['glm53-flash']
    assert server['depends_on'] == {'prepare': {'condition': 'service_completed_successfully'}}
    assert prepare['image'] == server['image'] == '${IMAGE:-' + release_image() + '}'
    assert prepare['entrypoint'] == ['/bin/bash', '/recipe/prepare.sh']
    assert server['entrypoint'] == ['/bin/bash', '/recipe/prepare.sh', 'serve']
    assert server['command'] == ['dflash'] and server['shm_size'] == '16g'
    assert server['env_file'] == ['docker/container.env']
    start = (ROOT / 'start.sh').read_text()
    for name in ('serve.sh', 'boot_check.py', 'p2p_check.py', 'p2p_check.sh', 'p2p_probe.cu'):
        assert f'./{name}:/recipe/{name}:ro' in server['volumes']
        assert f'-v "$SCRIPT_DIR/{name}:/recipe/{name}:ro"' in start
    assert './cache:/cache' in prepare['volumes'] and './cache:/cache' in server['volumes']
    assert '${MODELS_DIR:-./models}:/models' in prepare['volumes']
    assert 'HOST_UID' not in str(server)


def write_checkpoint(path, shards=('a.safetensors', 'b.safetensors')):
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    weight_map = ', '.join(f'"layer.{i}.weight": "{name}"' for i, name in enumerate(shards))
    (path / 'model.safetensors.index.json').write_text('{"weight_map": {' + weight_map + '}}')
    for name in shards:
        (path / name).write_bytes(b'x')
    return path


def check(path):
    return subprocess.run(['bash', str(PREPARE), 'check', str(path)]).returncode


def test_check_accepts_only_complete_checkpoints(tmp_path):
    assert check(write_checkpoint(tmp_path / 'complete')) == 0
    single = tmp_path / 'single'
    single.mkdir()
    (single / 'config.json').write_text('{}')
    assert check(single) == 1
    (single / 'model.safetensors').write_bytes(b'x')
    assert check(single) == 0
    assert check(tmp_path / 'absent') == 1
    shard = write_checkpoint(tmp_path / 'shard')
    (shard / 'b.safetensors').unlink()
    assert check(shard) == 1
    partial = write_checkpoint(tmp_path / 'partial')
    download = partial / '.cache' / 'huggingface' / 'download' / 'sub'
    download.mkdir(parents=True)
    (download / 'c.safetensors.incomplete').write_bytes(b'')
    assert check(partial) == 1
    noconfig = write_checkpoint(tmp_path / 'noconfig')
    (noconfig / 'config.json').unlink()
    assert check(noconfig) == 1


def prepare_functions(body, **env):
    """Run prepare.sh's functions with stat and chown replaced by recorders."""
    source = PREPARE.read_text().split('case "${1:-}" in', 1)[0]
    script = source + '''
stat() { case "$3" in *root*|*checkout*) echo "${FAKE_OWNER:-0}" ;; *) echo 1000 ;; esac; }
chown() { echo "chown $*"; }
''' + body
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                          env=dict(os.environ, **env))


def test_claim_changes_only_directories_docker_created(tmp_path):
    fresh = tmp_path / 'root-fresh'
    fresh.mkdir()
    (fresh / 'GLM-5.3-Flash-DFlash2').mkdir()
    other = tmp_path / 'root-other'
    other.mkdir()
    (other / 'lost+found').mkdir()
    user = tmp_path / 'user-owned'
    user.mkdir()
    run = prepare_functions(f'''SERVE_UID=1234; SERVE_GID=4321
claim "{fresh}" GLM-5.3-Flash-W4A16-MTP GLM-5.3-Flash-DFlash2
claim "{other}" GLM-5.3-Flash-W4A16-MTP GLM-5.3-Flash-DFlash2
claim "{user}"
''')
    assert run.returncode == 0, run.stderr
    assert f'chown 1234:4321 {fresh}' in run.stdout
    assert str(other) not in run.stdout and str(user) not in run.stdout


@pytest.mark.parametrize('env,expected', [({'FAKE_OWNER': '1001'}, '1001 1001'),
                                          ({'HOST_UID': '1500', 'HOST_GID': '1600'}, '1500 1600'),
                                          ({'FAKE_OWNER': '0'}, '0 0')])
def test_serving_user_is_the_checkout_owner(tmp_path, env, expected):
    (tmp_path / 'checkout').mkdir()
    run = prepare_functions(f'''CHECKOUT="{tmp_path}/checkout"
serving_ids; echo "$SERVE_UID $SERVE_GID"
as_serving_user false; echo returned
''', **{'HOST_UID': '', 'HOST_GID': '', **env})
    assert run.stdout.splitlines()[0] == expected, run.stderr
    assert ('returned' in run.stdout) == expected.startswith('0 ')


def test_rejects_non_numeric_override():
    run = prepare_functions('serving_ids', HOST_UID='ba', HOST_GID='')
    assert run.returncode == 1 and 'numeric' in run.stderr


@pytest.fixture
def checkout(tmp_path):
    (tmp_path / '.env.example').write_text('')
    (tmp_path / '.env').write_text('LAYOUT=tp4\n')
    (tmp_path / 'venv').mkdir()
    (tmp_path / 'venv' / '.recipe-stamp').write_text('old\n')
    (tmp_path / 'logs').mkdir()
    model = tmp_path / 'models' / 'GLM-5.3-Flash-W4A16-MTP'
    model.mkdir(parents=True)
    server = subprocess.Popen(['bash', '-c', f'exec -a "vllm serve {model}" sleep 120'],
                              start_new_session=True)
    (tmp_path / 'logs' / 'vllm.pid').write_text(str(server.pid))
    yield tmp_path, server
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGKILL)
    server.wait()


def run_start(root, body, **env):
    source = (ROOT / 'start.sh').read_text().rsplit('main "$@"', 1)[0]
    script = root / 'start.sh'
    script.write_text(source + body)
    return subprocess.run(['bash', str(script)], capture_output=True, text=True,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES='', **env), timeout=60)


def test_update_leaves_native_server_running_without_docker(checkout):
    root, server = checkout
    run = run_start(root, 'docker() { return 1; }\ndo_update\n')
    assert run.returncode == 1
    assert 'left running' in run.stderr and server.poll() is None
    assert (root / 'logs' / 'vllm.pid').exists()


def test_update_moves_native_install_to_container(checkout):
    root, server = checkout
    env_before = (root / '.env').read_bytes()
    run = run_start(root, '''docker() { return 0; }
STOP_TIMEOUT=10
start_unlocked() { echo "container started"; }
do_update
''')
    assert run.returncode == 0, run.stdout + run.stderr
    out = run.stdout
    assert 'this release runs the engine only in the container' in out
    assert f'native server (pid {server.pid}) is stopped' in out
    assert 'left in place' in out and out.index('left in place') < out.index('container started')
    server.wait(timeout=10)
    assert not (root / 'logs' / 'vllm.pid').exists()
    assert (root / 'venv' / '.recipe-stamp').exists() and (root / 'models').is_dir()
    assert (root / '.env').read_bytes() == env_before
    assert (root / 'logs' / 'native-retired').exists()
    again = run_start(root, 'docker() { return 0; }\nnative_notice\n')
    assert 'native install' not in again.stdout


def test_plain_start_refuses_while_native_server_runs(checkout):
    root, server = checkout
    run = run_start(root, 'take_lock() { :; }\nstart_unlocked\n')
    assert run.returncode == 1 and './start.sh update' in run.stderr
    time.sleep(0.2)
    assert server.poll() is None


def test_retired_native_settings_are_named(tmp_path):
    (tmp_path / '.env.example').write_text('')
    run = run_start(tmp_path, 'retired_settings\n', RUNTIME='native', VLLM_COMMIT='0' * 40,
                    VLLM_BUILD_AMPERE_MARLIN='1')
    for name in ('RUNTIME=native', 'VLLM_COMMIT=0000000000', 'VLLM_BUILD_AMPERE_MARLIN'):
        assert name in run.stderr
    quiet = run_start(tmp_path, 'retired_settings\n', RUNTIME='container')
    assert quiet.stderr == ''
