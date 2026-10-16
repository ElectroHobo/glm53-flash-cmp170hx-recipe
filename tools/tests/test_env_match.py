from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reference(tmp_path):
    values = {
        "tp4": {
            "MAX_BATCHED": "3463",
            "VLLM_AMPERE_MARLIN_DECODE_CUDA": "1",
            "VLLM_AMPERE_MARLIN_DECODE_V2": "1",
            "VLLM_AMPERE_KDA_FOLD": "1",
            "VLLM_CUSTOM_ALLREDUCE_FLAGS": "1",
            "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": "1",
            "NCCL_P2P_LEVEL": "SYS",
            "VLLM_THIN_GEMM": "1",
            "VLLM_MOE_MASK_PADDING": "1",
        },
        "pp4": {
            "MAX_BATCHED": "2312",
            "VLLM_PP_LAYER_PARTITION": "13,11,11,10",
            "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS": "9-64",
            "VLLM_AMPERE_MARLIN_PREFILL_WIDE": "1",
            "VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED": "1",
            "NCCL_P2P_LEVEL": "SYS",
            "VLLM_HOST_ALLREDUCE": "1",
            "VLLM_SHARED_EXPERT_REORDER": "1",
        },
    }
    for layout in values.values():
        layout.update({
            "VLLM_DFLASH_BOUNDARY_CACHE": "1",
            "VLLM_FLA_PIN_AUTOTUNE": "1",
            "VLLM_INDEXER_DECODE_RAW_K": "1",
            "VLLM_TOOL_CHOICE_NONE_MASK": "1",
        })
    return tmp_path / "reference.json", values


def run(reference):
    path, values = reference
    path.write_text(json.dumps(values))
    environment = dict(os.environ, VLLM_DFLASH_BOUNDARY_CACHE="0",
                       VLLM_AMPERE_DECODE_KERNELS="0")
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check-env-match.py"), str(path)],
        env=environment, text=True, capture_output=True,
    )


def test_defaults_match_without_inheriting_shell_overrides(reference):
    result = run(reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Environment match: PASS" in result.stdout


@pytest.mark.parametrize("layout,key,value", [
    ("tp4", "VLLM_THIN_GEMM", "0"),
    ("pp4", "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS", "9-32"),
    ("pp4", "VLLM_INDEXER_DECODE_RAW_K", "0"),
    ("tp4", "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", "0"),
    ("pp4", "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", "1"),
    ("tp4", "NCCL_P2P_LEVEL", "PIX"),
    ("pp4", "NCCL_P2P_LEVEL", "LOC"),
    ("pp4", "VLLM_AMPERE_KDA_FOLD", "1"),
    ("pp4", "VLLM_GLM5_MOE_MASK_PADDING", "1"),
])
def test_reports_changed_validation_value(reference, layout, key, value):
    reference[1][layout][key] = value
    result = run(reference)
    assert result.returncode == 1
    assert f"{layout}: {key}: expected" in result.stdout


def test_narrow_selector_input_is_inert_only_with_pp4_whole_expert_path(reference):
    reference[1]["pp4"]["VLLM_AMPERE_MARLIN_PREFILL_NARROW"] = "1"
    result = run(reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pp4: ignored VLLM_AMPERE_MARLIN_PREFILL_NARROW (validation layout input" in result.stdout
    reference[1]["pp4"]["VLLM_AMPERE_MARLIN_PREFILL_WIDE"] = "0"
    assert run(reference).returncode == 1
    reference[1]["pp4"]["VLLM_AMPERE_MARLIN_PREFILL_WIDE"] = "1"
    reference[1]["tp4"]["VLLM_AMPERE_MARLIN_PREFILL_NARROW"] = "0"
    assert run(reference).returncode == 1


def test_lists_diagnostic_and_path_exclusions(reference):
    reference[1]["tp4"].update({
        "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE": "diagnostic",
        "FLASHINFER_WORKSPACE_BASE": "/validation/cache",
    })
    result = run(reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ignored VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE (diagnostic only)" in result.stdout
    assert "ignored FLASHINFER_WORKSPACE_BASE (path valued)" in result.stdout
