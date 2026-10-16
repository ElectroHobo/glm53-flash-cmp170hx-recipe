#!/usr/bin/env python3
"""Compare CPU launcher dry runs with a supplied validation environment."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = {
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE",
}


def check(reference):
    failures = []
    metadata = sorted(key for key in reference if key.startswith("_"))
    print("Ignored metadata: " + (", ".join(metadata) or "none"))
    with tempfile.TemporaryDirectory(prefix="env-match-") as scratch:
        base = Path(scratch)
        (base / "bin").mkdir()
        (base / "bin/python").symlink_to(sys.executable)
        package = base / "vllm"
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "_ampere_marlin_C.py").touch()
        print("CPU installation fixture: optional Marlin module discoverable")
        for layout in ("tp4", "pp4"):
            expected = reference[layout]
            environment = {
                "PATH": os.defpath, "CUDA_VISIBLE_DEVICES": "", "DRY": "1",
                "LAYOUT": layout, "P2P": "force", "VENV": scratch, "PYTHONPATH": scratch,
            }
            result = subprocess.run(
                ["bash", str(ROOT / "serve.sh")], cwd=ROOT,
                env=environment, text=True, capture_output=True,
            )
            if result.returncode:
                failures.append(f"{layout}: dry run failed ({result.returncode})")
                continue
            actual = dict(re.findall(r"^  ([A-Z][A-Z0-9_]*)=(.*)$",
                                     result.stdout, re.M))
            command = shlex.split(result.stdout.split("DRY=1, command:\n", 1)[1])
            actual["MAX_BATCHED"] = command[command.index("--max-num-batched-tokens") + 1]
            print(f"{layout}: MAX_BATCHED checked in command; the speculative "
                  "configuration is compared in --effective mode")
            ignored = {}
            for key, value in expected.items():
                if key in DIAGNOSTICS:
                    ignored[key] = "diagnostic only"
                elif (key == "VLLM_AMPERE_MARLIN_PREFILL_NARROW" and layout == "pp4"
                        and str(value) == "1" and key not in actual
                        and actual.get("VLLM_AMPERE_MARLIN_PREFILL_WIDE") == "1"
                        and str(expected.get("VLLM_AMPERE_MARLIN_PREFILL_WIDE")) == "1"):
                    ignored[key] = ("validation layout input; PP4 whole experts use intermediate "
                                    "N=2048, not the TP4 selector's N=512 (see tools/env_effective.py)")
                elif isinstance(value, str) and value.startswith(("/", "~/")):
                    ignored[key] = "path valued"
                elif actual.get(key) != str(value):
                    failures.append(f"{layout}: {key}: expected {value!r}, "
                                    f"got {actual.get(key)!r}")
            for key, reason in sorted(ignored.items()):
                print(f"{layout}: ignored {key} ({reason})")
            extras = sorted(set(actual) - set(expected))
            print(f"{layout}: ignored exports unspecified by reference: "
                  + (", ".join(extras) or "none"))
            print(f"{layout}: compared {len(expected) - len(ignored)} settings")
    if failures:
        print("Environment match: FAIL\n" + "\n".join(failures))
        return 1
    print("Environment match: PASS (TP4 and PP4)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, nargs="?", help="validation JSON path")
    parser.add_argument("--effective", type=Path, help="full effective_env_private.json")
    parser.add_argument("--layout", choices=("tp4", "pp4"))
    parser.add_argument("--fork", type=Path,
                        default=Path.home() / "kernels/vllm-wt-bundle3-caaf",
                        help="fork checkout containing the pinned source objects")
    parser.add_argument("--serve-log", type=Path, help="default: serve.log beside effective JSON")
    args = parser.parse_args()
    if args.effective:
        if args.reference or not args.layout:
            parser.error("--effective requires --layout and cannot take reference")
        from env_effective import check_effective
        try:
            return check_effective(ROOT, args.effective, args.layout, args.fork, args.serve_log)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"Full environment match: FAIL ({exc})")
            return 1
    if not args.reference or args.layout or args.serve_log:
        parser.error("supply reference or --effective with --layout")
    return check(json.loads(args.reference.read_text()))


if __name__ == "__main__":
    sys.exit(main())
