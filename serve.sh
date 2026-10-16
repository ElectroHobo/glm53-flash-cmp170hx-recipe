#!/usr/bin/env bash
# Launch GLM-5.3-Flash W4A16 on 4x CMP 170HX from the patched vLLM checkout.
#
# Usage: serve.sh [dflash|mtp|none|--help]   (dflash is the only supported mode;
#   mtp and none are unsupported, untested: no validation, no issue support)
#   (no argument)  same as "dflash" -- the default.
#   dflash         DFlash2 drafter with the layout's validated adaptive depth:
#                  --speculative-config num_speculative_tokens 7 and adaptive_k
#                  (up to 7 at one request, 5 at two, 3 under load). SPEC_N is
#                  refused in this mode; pass your own --speculative-config in
#                  EXTRA_ARGS for another depth.
#   mtp            MTP drafter,     num_speculative_tokens = SPEC_N (default 3).
#   none           no speculative decoding.
# A --speculative-config inside EXTRA_ARGS wins over all of the above and no
# second one is added, so callers can pass their own JSON.
# Set DRY=1 to print the environment and the command that would run instead
# of running them.
#
# Paths (all relative to this repo unless overridden):
#   VENV            venv from install.sh            (default ./venv)
#   MODEL           target model directory          (default ./models/GLM-5.3-Flash-W4A16-MTP)
#   DFLASH_MODEL    drafter directory               (default ./models/GLM-5.3-Flash-DFlash2)
#   CUDA_HOME       CUDA toolkit root               (default /usr/local/cuda-13.3)
# API:
#   HOST            address to listen on (default 127.0.0.1: this machine only).
#                   0.0.0.0 listens on every interface -- set API_KEY first.
#   PORT            default 8000
#   API_KEY         if set, every /v1 request must send "Authorization: Bearer
#                   <key>" (passed to vLLM as VLLM_API_KEY, so it never appears
#                   on the command line). Unset = no key: anyone who can reach
#                   HOST:PORT can use the model.
#   SERVED_MODEL_NAME  model id clients send (default glm-5.3-flash; the older
#                   name SERVED_NAME is still read if this one is unset)
# Env overrides: LAYOUT, MAX_LEN, MAX_SEQS, MAX_BATCHED (default 3456 + draft
#   slots = 3463 under TP4, 3,456-token prefill chunks with the 7 draft slots;
#   2312 = 2,304-token chunks under PP4),
#   PREFILL_CAP (upstream long-prefill chunk cap, UNCONDITIONAL; default 0 = off
#   -- LEAVE IT 0, see below), GPU_UTIL, SPEC_N (mtp only), REASONING_PARSER,
#   TOOL_PARSER, MM_CAP, PP, TP, VLLM_PP_LAYER_PARTITION, BLOCK_SIZE, EXTRA_ARGS.
# Engine switches are exported only where the release value differs from the
# engine's code default; any other engine switch you set passes through.
#
# Optional compiled Marlin decode (on if installed, both layouts):
#   VLLM_AMPERE_MARLIN_DECODE_CUDA=0|1
#   The image includes the library.
#   Decode: eligible TP4/PP4 small batches, split-K reduction order.
#   Engine shape/token gates apply; no startup or layout-change compilation.
# Layout (LAYOUT):
#   tp4 (default)  tensor-parallel 4 (PP=1, TP=4). Fastest per request; one or
#                  two interactive users. Assumes PCIe Gen2 x16 links.
#   pp4            pipeline-parallel 4 (PP=4, TP=1). Each card holds a quarter
#                  of the layers and passes activations on: much faster
#                  prefill, about twice the KV, many parallel users, and far
#                  less traffic between the cards. Measured on x16 links.
#   An explicit PP/TP still wins over LAYOUT.
#
# ===== sm_80 feature flags ==================================================
# The performance features, the repeatable-output fixes and the KV headroom
# changes, plus an optional PCIe peer-to-peer switch, measured one by one and
# then together. All are off in the engine code; the release defaults below
# and the engine gates select them. Each feature has a one-variable switch --
# set it to 0 in the environment and restart, no rebuild and no revert.
# (Exception: VLLM_SPARSE_INDEXER_MAX_LOGITS_MB is off at 512.)
#
#   VLLM_GLM5_PREFILL_KERNELS=0   sm_80 prefill kernels
#   EXTRA_ARGS=--no-enable-batch-sharded-sampling
#                                 batch-sharded logits + sampling (TP only)
#   VLLM_AMPERE_DECODE_KERNELS=0  sm_80 decode kernels
#   VLLM_THIN_GEMM=0              sm_80 thin-M BF16 GEMM
#   FAIR_PREFILL=0                decode-aware prefill chunking
#   VLLM_HOST_ALLREDUCE=0         host-staged no-P2P all-reduce
#   VLLM_SHARED_EXPERT_REORDER=0  MoE shared experts after routed dispatch
#   VLLM_AMPERE_DECODE_IDX_GLUE=0 / VLLM_MOE_DECODE_ROUTE_V2=0 /
#   VLLM_GLM5_DECODE_MHC_V2=0 /
#   VLLM_GLM5_DRAFTER_ROPE_FIT=0  second-generation decode, both layouts
#   VLLM_AMPERE_KDA_FOLD=0 / VLLM_THIN_GEMM_CONCAT=0 /
#   VLLM_AMPERE_KDA_SMALL_STACK=0 TP4 decode launch reductions, LAYOUT=tp4 only
#   VLLM_AMPERE_MARLIN_DECODE_V2=0  compiled Marlin decode v2, LAYOUT=tp4 only
#   VLLM_MOE_DETERMINISTIC_ALIGN=0 / VLLM_MOE_MASK_PADDING=0 /
#   VLLM_GLM5_TOPK_TIEFIX=0 / VLLM_GLM5_TOPK_SORTED=0 (needs TOPK_TIEFIX=1) /
#   VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=0 / VLLM_FLA_PIN_AUTOTUNE=0
#                                 repeatable output
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 / VLLM_GLM5_DRAFTER_SELECTOR_SHARD=0
#                                 KV headroom (the selector shard is TP only)
#   VLLM_INDEXER_DECODE_RAW_K=0   raw-bit indexer K dequant, both layouts
#   VLLM_TOOL_CHOICE_NONE_MASK=0  tool_choice=none decode mask, both layouts
#   VLLM_KV_MAMBA_INFLIGHT_STATES=0  KV accounting for prefill chunks in flight,
#                                 both layouts (0 reports the old, larger pool)
#   VLLM_GLM5_SMLA_PREFILL_PRED_LOAD=0  sparse-attention prefill gather that
#                                 skips empty slots, both layouts
#   VLLM_AMPERE_KDA_PREFILL_NARROW=0 / VLLM_AMPERE_MARLIN_PREFILL_NARROW=0
#                                 TP4 prefill kernels, LAYOUT=tp4 only
#   VLLM_AMPERE_KDA_PREFILL_WIDE=0 / VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE=0 /
#   VLLM_AMPERE_MARLIN_PREFILL_WIDE=0 / VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED=0
#                                 PP4 prefill kernels, LAYOUT=pp4 only
#   VLLM_PP_DRAFT_TAIL_STAGE=-1   PP4 drafter's final step on stage 2, PP only
#   VLLM_PP_SPREAD_DECODES=0 / VLLM_PP_PACKED_HOP=0 / VLLM_PP_HOP_NO_METADATA=0 /
#   VLLM_PP_SPLIT_DRAFT_EVENT=0 / VLLM_GLM5_PP_FOLD_DRAFT_FC=0
#                                 PP4 pipeline and drafter changes, PP only
#   P2P=auto                      verify peer content before enabling CUDA/NCCL P2P
#   P2P=off                       disable all peer transports (also PP)
#   P2P=force                     bypass the content check
#   VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0  disable custom all-reduce after a pass (TP)
#   GLM5_NCCL_P2P_SYS=0           NCCL over peer-to-peer too (NCCL_P2P_LEVEL=SYS);
#                                 on by default after a content-check pass, under
#                                 TP while the custom all-reduce is on, and under PP.
#   MAX_BATCHED=2048              1,152-token prefill chunks, as 1.2.0 shipped
#                                 (the TP4 default 3463 gives 3,456-token chunks).
#
# If output quality is ever in question, try VLLM_AMPERE_DECODE_KERNELS=0 first.
#
# PREFILL_CAP MUST STAY 0. It is upstream's UNCONDITIONAL chunk cap: it cost
# -15.3% prefill / +16.5% time to first token on a 23K prompt here, and as a
# side effect it drops chunks below the two prefill gates and silently
# disables the prefill kernels. The decode-aware cap (FAIR_PREFILL) is the one
# to use -- it only applies while something is actually decoding.
# ============================================================================
set -euo pipefail
# generated:serving-defaults:begin
# Serving defaults for GLM-5.3-Flash; recipe copies are generated.
# Constants are deliberately not exported. Override the existing launcher knobs,
# not these values. TP4/PP4 selection and unset-versus-empty policy stay explicit
# in the launchers. An engine switch is listed only under its current name and
# only where its serving value differs from the code default (a launcher leaves
# a switch at its default unset); the key is GLM_DEFAULT_<switch>, with a _PP4
# or _TP4 suffix where the value belongs to one layout.

GLM_DEFAULT_BLOCK_SIZE=4608
GLM_DEFAULT_BOOT_CHECK=1
GLM_DEFAULT_FAIR_CHUNK=384
GLM_DEFAULT_FAIR_PARTIAL=2
GLM_DEFAULT_FAIR_PREFILL=1
GLM_DEFAULT_GPU_UTIL=0.95
GLM_DEFAULT_HOST=127.0.0.1
GLM_DEFAULT_LAYOUT=tp4
GLM_DEFAULT_MAX_LEN=262144
GLM_DEFAULT_MAX_SEQS=8
GLM_DEFAULT_MM_CAP=0
GLM_DEFAULT_MM_FRAMES=32
GLM_DEFAULT_MM_IMAGES=4
GLM_DEFAULT_MM_MAX_PIXELS=1003520
GLM_DEFAULT_NCCL_P2P_LEVEL=SYS
GLM_DEFAULT_PORT=8000
GLM_DEFAULT_PP_PP4=4
GLM_DEFAULT_PP_TP4=1
GLM_DEFAULT_PREFILL_CAP=0
GLM_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
GLM_DEFAULT_READY_TIMEOUT=1800
GLM_DEFAULT_REASONING_PARSER=glm47
GLM_DEFAULT_SPEC_N=3
GLM_DEFAULT_TOOL_PARSER=glm47
GLM_DEFAULT_TP_PP4=1
GLM_DEFAULT_TP_TP4=4
GLM_DEFAULT_VLLM_AMPERE_DECODE_KERNELS=1
GLM_DEFAULT_VLLM_AMPERE_KDA_FOLD=1
GLM_DEFAULT_VLLM_AMPERE_KDA_PREFILL_NARROW=1
GLM_DEFAULT_VLLM_AMPERE_KDA_PREFILL_WIDE=1
GLM_DEFAULT_VLLM_AMPERE_KDA_SMALL_STACK=1
GLM_DEFAULT_VLLM_AMPERE_MARLIN_DECODE_CUDA=1
GLM_DEFAULT_VLLM_AMPERE_MARLIN_DECODE_V2=1
GLM_DEFAULT_VLLM_AMPERE_MARLIN_PREFILL_NARROW=1
GLM_DEFAULT_VLLM_AMPERE_MARLIN_PREFILL_WIDE=1
GLM_DEFAULT_VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED_PP4=1
GLM_DEFAULT_VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE=1
GLM_DEFAULT_VLLM_CUSTOM_ALLREDUCE_ALGO=2stage
GLM_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS=1
GLM_DEFAULT_VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1
GLM_DEFAULT_VLLM_DFLASH_BOUNDARY_CACHE=1
GLM_DEFAULT_VLLM_FLA_PIN_AUTOTUNE=1
GLM_DEFAULT_VLLM_GLM5_DRAFTER_SELECTOR_SHARD=1
GLM_DEFAULT_VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS=1
GLM_DEFAULT_VLLM_GLM5_MARLIN_DECODE_PP_MULTI=1
GLM_DEFAULT_VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS_PP4=9-64
GLM_DEFAULT_VLLM_GLM5_PP_FOLD_DRAFT_FC=1
GLM_DEFAULT_VLLM_GLM5_PREFILL_KERNELS=1
GLM_DEFAULT_VLLM_GLM5_PREFILL_MIN_TOKENS=384
GLM_DEFAULT_VLLM_GLM5_SMLA_PREFILL_PRED_LOAD=1
GLM_DEFAULT_VLLM_GLM5_TOPK_SORTED=1
GLM_DEFAULT_VLLM_GLM5_TOPK_TIEFIX=1
GLM_DEFAULT_VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=8
GLM_DEFAULT_VLLM_HOST_ALLREDUCE=1
GLM_DEFAULT_VLLM_INDEXER_DECODE_RAW_K=1
GLM_DEFAULT_VLLM_KV_MAMBA_INFLIGHT_STATES=1
GLM_DEFAULT_VLLM_MOE_DETERMINISTIC_ALIGN=1
GLM_DEFAULT_VLLM_MOE_MASK_PADDING=1
GLM_DEFAULT_VLLM_PP_DRAFT_TAIL_STAGE=2
GLM_DEFAULT_VLLM_PP_HOP_NO_METADATA=1
GLM_DEFAULT_VLLM_PP_PACKED_HOP=1
GLM_DEFAULT_VLLM_PP_SPLIT_DRAFT_EVENT=1
GLM_DEFAULT_VLLM_PP_SPREAD_DECODES=1
GLM_DEFAULT_VLLM_SHARED_EXPERT_REORDER=1
GLM_DEFAULT_VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
GLM_DEFAULT_VLLM_THIN_GEMM=1
GLM_DEFAULT_VLLM_THIN_GEMM_CONCAT=1
GLM_DEFAULT_VLLM_TOOL_CHOICE_NONE_MASK=1

# Layout geometry and shared argv defaults.
GLM_DEFAULT_MODE=dflash
GLM_DEFAULT_PART_DFLASH=13,11,11,10
GLM_DEFAULT_PART_MTP=14,12,12,7
GLM_DEFAULT_TP4V2=1
GLM_DEFAULT_CHUNK_TP4=3456
GLM_DEFAULT_MIN_DRAFT_SLOTS=4
GLM_DEFAULT_MAX_BATCHED_PP4=2312
GLM_DEFAULT_SERVED_MODEL_NAME=glm-5.3-flash
GLM_DEFAULT_MM_WIDTH=1280
GLM_DEFAULT_MM_HEIGHT=784
GLM_DEFAULT_TRUST_REMOTE_CODE=--trust-remote-code
GLM_DEFAULT_AUTO_TOOL_CHOICE=--enable-auto-tool-choice
# DFlash adaptive depth: --speculative-config num_speculative_tokens and the
# per-layout adaptive_k mapping (by_load depths, measured depth-3..7 accept costs).
GLM_DEFAULT_DFLASH_DEPTH=7
GLM_DEFAULT_ADAPTIVE_K_TP4='{"by_load":[7,5,3],"draft_by_load":true,"accept":{"costs":[1.0,1.1009,1.1631,1.2214,1.2718],"costs_multi":[1.0,1.1132,1.1958,1.2214,1.2718],"hysteresis":0.03},"log_interval":0}'
GLM_DEFAULT_ADAPTIVE_K_PP4='{"by_load":[7,5,3],"accept":{"costs":[1.0,1.0894,1.1375,1.2001,1.2239],"costs_multi":[1.0,1.0889,1.1194,1.2001,1.2239],"hysteresis":0.03},"log_interval":0}'
# generated:serving-defaults:end
MODE=${1:-$GLM_DEFAULT_MODE}
case "$MODE" in
  -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
  dflash|mtp|none) ;;
  *) echo "serve.sh: unknown argument '$MODE' (expected dflash, mtp, none or --help)" >&2; exit 2 ;;
esac
# Container compilers use a private temporary directory on the cache mount.
# This runs for start.sh, Compose and direct launches before Python imports.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    [ "$RECIPE_CACHE_ROOT" = /cache ] || { echo '[boot-check] FAIL: container cache root must be /cache' >&2; exit 1; }
    temp_base="${TMPDIR:-/cache/tmp}"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR must be under /cache' >&2; exit 1 ;; esac
    mkdir -p "$temp_base"
    temp_base="$(realpath "$temp_base")"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR resolves outside /cache' >&2; exit 1 ;; esac
    TMPDIR="$(mktemp -d "$temp_base/run.XXXXXXXX")"
    export TMPDIR
    export TMP="$TMPDIR" TEMP="$TMPDIR"
    unset temp_base
fi
# Cache seeds are opt-in until the image's cold/warm acceptance passes.
if [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ] && [ "${DRY:-0}" = 1 ]; then
    echo "[image-cache] enabled (dry run)"
elif [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ]; then
    [ -f /opt/image-tools/cache_seed.py ] || { echo "[image-cache] enabled but image helper missing" >&2; exit 1; }
    cache_exports="$(/opt/venv/bin/python /opt/image-tools/cache_seed.py seed /opt/cache-seed)"
    eval "$cache_exports"
    unset cache_exports
else
    echo "[image-cache] disabled"
fi

# Explicit compiler destinations also cover launches without a seed. The custom
# all-reduce and host-staged all-reduce builds go under VLLM_CACHE_ROOT and
# TORCH_EXTENSIONS_DIR.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    export HOME="${HOME:-/cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    export HF_HOME="${HF_HOME:-/cache/huggingface}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$HOME/.tilelang/cache/tmp}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/torchinductor}"
    for cache_path in "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT" "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR" "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR"; do
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: compiler cache outside /cache: $cache_path" >&2; exit 1 ;; esac
        mkdir -p "$cache_path"
        cache_path="$(realpath "$cache_path")"
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: cache symlink outside /cache: $cache_path" >&2; exit 1 ;; esac
        [ -w "$cache_path" ] || { echo "[boot-check] FAIL: cache is not writable: $cache_path" >&2; exit 1; }
    done
    unset cache_path
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
MODEL="${MODEL:-$REPO_ROOT/models/GLM-5.3-Flash-W4A16-MTP}"
DFLASH_MODEL="${DFLASH_MODEL:-$REPO_ROOT/models/GLM-5.3-Flash-DFlash2}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

# Layout. LAYOUT=tp4 (default): TENSOR-PARALLEL 4. Every layer is split across
# all four cards, so it is the fastest per request, but it assumes PCIe Gen2 x16
# links: it moves ~9.4 MB per layer during prefill and ~100 small collectives
# per decode step, so on stock x4 links it is bus-bound.
# LAYOUT=pp4: PIPELINE-PARALLEL 4 (PP=4, TP=1). Each card holds a quarter of the
# layers and hands only activations to the next one, so it needs far less link
# bandwidth; it prefills much faster and holds about twice the KV, and it suits
# many parallel users rather than one fast one. It runs with the DFlash2
# drafter like TP4. Its numbers were measured on x16 links.
# An explicit PP/TP in the environment still wins over LAYOUT.
LAYOUT=${LAYOUT:-$GLM_DEFAULT_LAYOUT}
case "$LAYOUT" in
  tp4) ;;
  pp4) PP=${PP:-$GLM_DEFAULT_PP_PP4}; TP=${TP:-$GLM_DEFAULT_TP_PP4} ;;
  *) echo "serve.sh: unknown LAYOUT '$LAYOUT' (expected tp4 or pp4)" >&2; exit 2 ;;
esac
PP=${PP:-$GLM_DEFAULT_PP_TP4}; TP=${TP:-$GLM_DEFAULT_TP_TP4}   # PP*TP must be 4
# Under PP the balanced layer split is 3 dense + 42 MoE (~3.8 GiB each). MTP keeps a 13.8 GiB BF16
# draft layer on the last stage, so the balanced split differs by mode. DFlash's drafter KV rides the
# MLA tensors, so it uses the non-MTP split.
if [ "$PP" = "4" ]; then
  if [ "$MODE" = "mtp" ]; then DEFAULT_PART=$GLM_DEFAULT_PART_MTP; else DEFAULT_PART=$GLM_DEFAULT_PART_DFLASH; fi
  export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-$DEFAULT_PART}"
elif [ -n "${VLLM_PP_LAYER_PARTITION:-}" ]; then export VLLM_PP_LAYER_PARTITION; else unset VLLM_PP_LAYER_PARTITION; fi
# KV block size under PP with DFlash2. Each stage holds the full 64-head
# linear-attention (KDA) state, which sets the attention block to 4,480 tokens;
# the drafter's matching block would then be 1,120 tokens, not a multiple of 64,
# which pushes the drafter out of the shared KV layout and makes the last stage
# pay about 6x per block. 4608 is the next multiple of 256: drafter block 1,152,
# state page padded 5%, valid for SPEC_N up to 8.
BLOCK_ARGS=()
if [ "$PP" -gt 1 ] && [ "$MODE" = "dflash" ]; then BLOCK_ARGS=(--block-size "${BLOCK_SIZE:-$GLM_DEFAULT_BLOCK_SIZE}"); fi

# --- pipeline-parallel (PP4) --------------------------------------------------
# Stage 2 runs the drafter's final step. Outputs identical; paired PP4 runs
# gained 7.1% at four users and 3.8% at eight. Kill switch: -1.
if [ "$PP" -gt 1 ]; then
  export VLLM_PP_DRAFT_TAIL_STAGE=${VLLM_PP_DRAFT_TAIL_STAGE:-$GLM_DEFAULT_VLLM_PP_DRAFT_TAIL_STAGE}
  echo "serve.sh: pipeline: DRAFT_TAIL_STAGE=$VLLM_PP_DRAFT_TAIL_STAGE"
fi

# Pipeline and drafter changes, PP only, each a kill switch at 0:
#   VLLM_PP_SPREAD_DECODES     spread decoding requests over all in-flight
#                              micro-batches, so no stage idles while one
#                              micro-batch carries every decode
#   VLLM_PP_PACKED_HOP         stage-to-stage hand-off packed into one transfer
#   VLLM_PP_HOP_NO_METADATA    hand-off without the per-step metadata exchange
#   VLLM_PP_SPLIT_DRAFT_EVENT  finer synchronisation of the drafter's inputs
#   VLLM_GLM5_PP_FOLD_DRAFT_FC drafter input projection folded
# Outputs identical with and without the first four; the fold is checked
# against a 64-bit reference. Measured together against PP4 without them:
# 4-user aggregate +7.3%, cold prefill +2.0%, 1-user step -0.8%, 8,366 fewer KV
# tokens.
if [ "$PP" -gt 1 ]; then
  export VLLM_PP_SPREAD_DECODES=${VLLM_PP_SPREAD_DECODES:-$GLM_DEFAULT_VLLM_PP_SPREAD_DECODES}
  export VLLM_PP_PACKED_HOP=${VLLM_PP_PACKED_HOP:-$GLM_DEFAULT_VLLM_PP_PACKED_HOP}
  export VLLM_PP_HOP_NO_METADATA=${VLLM_PP_HOP_NO_METADATA:-$GLM_DEFAULT_VLLM_PP_HOP_NO_METADATA}
  export VLLM_PP_SPLIT_DRAFT_EVENT=${VLLM_PP_SPLIT_DRAFT_EVENT:-$GLM_DEFAULT_VLLM_PP_SPLIT_DRAFT_EVENT}
  export VLLM_GLM5_PP_FOLD_DRAFT_FC=${VLLM_GLM5_PP_FOLD_DRAFT_FC:-$GLM_DEFAULT_VLLM_GLM5_PP_FOLD_DRAFT_FC}
  echo "serve.sh: pipeline: SPREAD=$VLLM_PP_SPREAD_DECODES PACKED_HOP=$VLLM_PP_PACKED_HOP HOP_NO_METADATA=$VLLM_PP_HOP_NO_METADATA SPLIT_DRAFT_EVENT=$VLLM_PP_SPLIT_DRAFT_EVENT FOLD_DRAFT_FC=$VLLM_GLM5_PP_FOLD_DRAFT_FC"
fi
# --- prefill kernels, per layout ----------------------------------------------
# Sparse-attention prefill gather, both layouts: empty top-k slots fetch
# nothing instead of all reading the same cache row. Bitwise-identical output.
# PP4 (64 heads): 450 -> 356 ms over the sparse-attention layers of an
# 8.2K-token prompt; TP4 (16 heads): the first chunk's kernel 1.58x faster.
# Kill switch: 0.
export VLLM_GLM5_SMLA_PREFILL_PRED_LOAD=${VLLM_GLM5_SMLA_PREFILL_PRED_LOAD:-$GLM_DEFAULT_VLLM_GLM5_SMLA_PREFILL_PRED_LOAD}
# PP4 prefill kernels (LAYOUT=pp4 only; each gated in the engine to 64 heads /
# whole experts on sm_80): linear-attention (KDA) chunked prefill, a new
# sparse-attention prefill kernel and a split-block Marlin MoE prefill, whose
# compiled prefill tiles (VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED, off in the
# engine code) are switched on here. Each is at least as accurate as the code
# it replaces against a 64-bit reference on real inputs. With the gather
# above: cold prefill +19.5%. Kill switches: 0.
if [ "$LAYOUT" = pp4 ]; then
  export VLLM_AMPERE_KDA_PREFILL_WIDE=${VLLM_AMPERE_KDA_PREFILL_WIDE:-$GLM_DEFAULT_VLLM_AMPERE_KDA_PREFILL_WIDE}
  export VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE=${VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE:-$GLM_DEFAULT_VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE}
  export VLLM_AMPERE_MARLIN_PREFILL_WIDE=${VLLM_AMPERE_MARLIN_PREFILL_WIDE:-$GLM_DEFAULT_VLLM_AMPERE_MARLIN_PREFILL_WIDE}
  export VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED=${VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED:-$GLM_DEFAULT_VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED_PP4}
  echo "serve.sh: PP4 prefill kernels: KDA=$VLLM_AMPERE_KDA_PREFILL_WIDE SPARSE_MLA=$VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE MARLIN=$VLLM_AMPERE_MARLIN_PREFILL_WIDE COMPILED=$VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED PRED_LOAD=$VLLM_GLM5_SMLA_PREFILL_PRED_LOAD"
fi
# TP4 prefill kernels (LAYOUT=tp4 only; gated in the engine to sm_80): the
# KDA chunked prefill at 16 heads (1.42x per prompt per card) and the
# split-block Marlin MoE prefill at TP4 shards (1.24x; from 384 tokens,
# VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS). Accuracy against a 64-bit reference
# at least equal to the code they replace. Kill switches: 0.
if [ "$LAYOUT" = tp4 ]; then
  export VLLM_AMPERE_KDA_PREFILL_NARROW=${VLLM_AMPERE_KDA_PREFILL_NARROW:-$GLM_DEFAULT_VLLM_AMPERE_KDA_PREFILL_NARROW}
  export VLLM_AMPERE_MARLIN_PREFILL_NARROW=${VLLM_AMPERE_MARLIN_PREFILL_NARROW:-$GLM_DEFAULT_VLLM_AMPERE_MARLIN_PREFILL_NARROW}
  echo "serve.sh: TP4 prefill kernels: KDA=$VLLM_AMPERE_KDA_PREFILL_NARROW MARLIN=$VLLM_AMPERE_MARLIN_PREFILL_NARROW PRED_LOAD=$VLLM_GLM5_SMLA_PREFILL_PRED_LOAD"
fi
# TP4 decode launch reductions (LAYOUT=tp4 with PP=1 only, the layout they
# were validated in): the linear-attention (KDA) state commit folded into the
# next verify, the attention thin-GEMM concat groups as one launch each (needs
# VLLM_THIN_GEMM) and the small-stack KDA verify configuration. Bitwise
# identical by design. Kill switches: 0.
if [ "$LAYOUT" = tp4 ] && [ "$PP" = 1 ]; then
  export VLLM_AMPERE_KDA_FOLD=${VLLM_AMPERE_KDA_FOLD:-$GLM_DEFAULT_VLLM_AMPERE_KDA_FOLD}
  export VLLM_THIN_GEMM_CONCAT=${VLLM_THIN_GEMM_CONCAT:-$GLM_DEFAULT_VLLM_THIN_GEMM_CONCAT}
  export VLLM_AMPERE_KDA_SMALL_STACK=${VLLM_AMPERE_KDA_SMALL_STACK:-$GLM_DEFAULT_VLLM_AMPERE_KDA_SMALL_STACK}
  echo "serve.sh: TP4 decode launches: KDA_FOLD=$VLLM_AMPERE_KDA_FOLD THIN_GEMM_CONCAT=$VLLM_THIN_GEMM_CONCAT KDA_SMALL_STACK=$VLLM_AMPERE_KDA_SMALL_STACK"
fi

# Release peer collective defaults; explicit overrides win.
# PP4 whole-expert compiled decode: one-request depths 4..6 and 9..64 rows.
# These switches require the optional decode library; without PP they stay at
# the engine defaults (off).
if [ "$PP" -gt 1 ]; then
  if [ -z "${VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS+x}" ]; then export VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS=$GLM_DEFAULT_VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS; fi
  if [ -z "${VLLM_GLM5_MARLIN_DECODE_PP_MULTI+x}" ]; then export VLLM_GLM5_MARLIN_DECODE_PP_MULTI=$GLM_DEFAULT_VLLM_GLM5_MARLIN_DECODE_PP_MULTI; fi
  if [ -z "${VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS+x}" ]; then export VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS=$GLM_DEFAULT_VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS_PP4; fi
else
  if [ -z "${VLLM_CUSTOM_ALLREDUCE_FLAGS+x}" ]; then export VLLM_CUSTOM_ALLREDUCE_FLAGS=$GLM_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS; fi
fi
echo "serve.sh: [release-features] all_reduce_flags=${VLLM_CUSTOM_ALLREDUCE_FLAGS:-engine default} pp_marlin_mid_rows=${VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS:-engine default} pp_marlin_multi=${VLLM_GLM5_MARLIN_DECODE_PP_MULTI:-engine default} pp_marlin_multi_rows=${VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS:-engine default}"

# --- KV accounting --------------------------------------------------------------
# With prefill chunks in flight, a request can hold more linear-attention state
# blocks than the engine used to reserve for it, so the KV pool it reported was
# larger than it can actually hold. This setting makes the reserve match, in
# both layouts. It changes the reported pool and the start-up fit check only;
# outputs cannot change.
#   VLLM_KV_MAMBA_INFLIGHT_STATES  linear-attention states held by chunks in
#                                  flight (TP4: 17,932 fewer tokens reported,
#                                  -1.5%)
# 0 brings back the old, larger figure. Kill switch: 0.
export VLLM_KV_MAMBA_INFLIGHT_STATES=${VLLM_KV_MAMBA_INFLIGHT_STATES:-$GLM_DEFAULT_VLLM_KV_MAMBA_INFLIGHT_STATES}
echo "serve.sh: KV accounting: MAMBA_INFLIGHT_STATES=$VLLM_KV_MAMBA_INFLIGHT_STATES"

# Independent compiled paths: decode defaults on in both layouts when installed.
# Discover the module file without importing vllm, torch or the extension.
# Explicit values (including invalid empty values) are never replaced.
if [ "${VLLM_AMPERE_MARLIN_DECODE_CUDA+x}" != x ]; then
  VLLM_AMPERE_MARLIN_DECODE_CUDA=0
  if { [ "$PP" = 4 ] && [ "$TP" = 1 ]; } || { [ "$TP" = 4 ] && [ "$PP" = 1 ]; }; then
   if [ "${DRY:-0}" = 1 ] && [ ! -x "$VENV/bin/python" ]; then
    # A dry run on a host has no engine to look into; show the default the
    # image gives (its build requires the optional library).
    VLLM_AMPERE_MARLIN_DECODE_CUDA=$GLM_DEFAULT_VLLM_AMPERE_MARLIN_DECODE_CUDA
    echo "serve.sh: [ampere-marlin] DRY=1 without the engine at $VENV: showing the image default"
   else
    VLLM_AMPERE_MARLIN_DECODE_CUDA="$(CUDA_VISIBLE_DEVICES= "$VENV/bin/python" - <<'PY'
import importlib.machinery
import importlib.util

package = importlib.util.find_spec("vllm")
locations = package.submodule_search_locations if package is not None else None
extension = (importlib.machinery.PathFinder.find_spec("vllm._ampere_marlin_C", locations)
             if locations is not None else None)
print("1" if extension is not None else "0")
PY
    )" || { echo "serve.sh: cannot determine optional Marlin installation" >&2; exit 1; }
    if [ "$VLLM_AMPERE_MARLIN_DECODE_CUDA" = 1 ]; then
      VLLM_AMPERE_MARLIN_DECODE_CUDA=$GLM_DEFAULT_VLLM_AMPERE_MARLIN_DECODE_CUDA
    else
      echo "serve.sh: [ampere-marlin] optional library absent; unset decode defaults OFF"
    fi
   fi
  fi
fi
export VLLM_AMPERE_MARLIN_DECODE_CUDA
case "$VLLM_AMPERE_MARLIN_DECODE_CUDA" in
  0|1) ;;
  *) echo "serve.sh: VLLM_AMPERE_MARLIN_DECODE_CUDA must be 0 or 1" >&2; exit 2 ;;
esac
# Compiled Marlin MoE decode v2 in place of the first compiled decode kernel
# (LAYOUT=tp4 with PP=1 only; the PP4 shapes keep the first kernel). Needs
# VLLM_AMPERE_MARLIN_DECODE_CUDA=1. Kill switch: 0.
if [ "$LAYOUT" = tp4 ] && [ "$PP" = 1 ]; then
  export VLLM_AMPERE_MARLIN_DECODE_V2=${VLLM_AMPERE_MARLIN_DECODE_V2:-$GLM_DEFAULT_VLLM_AMPERE_MARLIN_DECODE_V2}
fi
echo "serve.sh: [ampere-marlin] decode=$VLLM_AMPERE_MARLIN_DECODE_CUDA decode_v2=${VLLM_AMPERE_MARLIN_DECODE_V2:-engine default} (split-K order; prebuilt library required when enabled)"
if [ "${DRY:-0}" != 1 ] && [ "$VLLM_AMPERE_MARLIN_DECODE_CUDA" = 1 ]; then
  CUDA_VISIBLE_DEVICES= "$VENV/bin/python" - <<'PY' || exit 1
try:
    from vllm.sm80.kernels.marlin_moe.extension import require_extension
    require_extension()
except (ImportError, RuntimeError, OSError) as exc:
    raise SystemExit("Optional Marlin requested but unavailable: use the release image, "
                     "which includes the compatible optional library, or set "
                     "VLLM_AMPERE_MARLIN_DECODE_CUDA=0. Details: " + str(exc))
PY
fi

# --- sm_80 feature flags (see the header for the kill switches) -------------
# Prefill kernels: sm_80 mHC pre-norm projection + sparse-MLA DSA attention.
export VLLM_GLM5_PREFILL_KERNELS=${VLLM_GLM5_PREFILL_KERNELS:-$GLM_DEFAULT_VLLM_GLM5_PREFILL_KERNELS}
# 384, not the code default 512: FAIR_PREFILL caps the chunk at 384 while
# something decodes, so a 512 gate switches these kernels off exactly then.
# Measured: contended 20K TTFT 12.98 -> 11.88 s, lone prompt flat.
export VLLM_GLM5_PREFILL_MIN_TOKENS=${VLLM_GLM5_PREFILL_MIN_TOKENS:-$GLM_DEFAULT_VLLM_GLM5_PREFILL_MIN_TOKENS}
# Batch-sharded logits: sample a 1/TP slice of the batch per rank instead of
# all-gathering full-vocab logits everywhere. vLLM's
# --enable-batch-sharded-sampling, passed under TP only: with TP=1 there is
# nothing to shard and the option stays at its default (off). Banner
# "Batch-sharded sampling enabled." Kill switch:
# EXTRA_ARGS=--no-enable-batch-sharded-sampling (EXTRA_ARGS comes last).
SAMPLING_ARGS=()
if [ "$TP" -gt 1 ]; then SAMPLING_ARGS=(--enable-batch-sharded-sampling); fi
echo "serve.sh: batch-sharded sampling=$([ "$TP" -gt 1 ] && echo on || echo off)"
# Decode kernels: sm_80 mHC fused post+pre, MoE routing/align, KDA decode.
# The per-family switches and token bounds keep their committed defaults --
# in particular the MoE padding-mask cutoff VLLM_MOE_MASK_PADDING_MAX_TOKENS
# (8). Do not set them here.
export VLLM_AMPERE_DECODE_KERNELS=${VLLM_AMPERE_DECODE_KERNELS:-$GLM_DEFAULT_VLLM_AMPERE_DECODE_KERNELS}
# Thin GEMM: sm_80 thin-M BF16 GEMM for the layers W4A16 leaves unquantized.
# Bounded at M <= 32 (VLLM_THIN_GEMM_MAX_TOKENS), which is a cudagraph
# capture size -- re-measure before changing it.
export VLLM_THIN_GEMM=${VLLM_THIN_GEMM:-$GLM_DEFAULT_VLLM_THIN_GEMM}
# Host-staged all-reduce for the TP decode collectives. Stock cards have no GPU
# peer access, so CustomAllreduce disables itself and NCCL's SHM ring pays
# 2(N-1) sequential host hops per message. The host-staged path does it in one
# round trip through a shared /dev/shm segment: measured 2.09x per all-reduce
# call in a decode trace (78.6 -> 37.6 us mean), decode step 19.10 -> 17.63 ms
# at 1 user and 37.6 -> 31.8 ms at 4 (alternating off/on restarts).
# Messages above 512 KiB stay on NCCL (VLLM_HOST_ALLREDUCE_MAX_SIZE) --
# prefill already runs near PCIe wire speed on the ring. Also gains 4,033 KV
# tokens. It stands aside by itself while a peer-to-peer CustomAllreduce
# serves, so it is what serves when VLLM_CUSTOM_ALLREDUCE_PCIE_P2P is 0.
# Kill switch: set it to 0 and restart.
# Set in both layouts, as validated: under PP4 there is no tensor-parallel
# all-reduce for it to replace, so it has nothing to do there.
export VLLM_HOST_ALLREDUCE=${VLLM_HOST_ALLREDUCE:-$GLM_DEFAULT_VLLM_HOST_ALLREDUCE}
# P2P=auto (default) enables peer access only after the startup content check.
# P2P=off disables both CUDA custom all-reduce and NCCL P2P, including PP.
# P2P=force bypasses verification. An explicit custom-all-reduce 0 is preserved
# after a pass. Use P2P=off to disable NCCL peer transfers too.
# 2stage, not the built-in crossover: that crossover takes one-shot below
# 512 KiB, which is tuned for NVLink and wrong on Gen2 x16. Forcing 1stage
# everywhere measured worse than either (16.83 ms at 1 user, 33.00 at 4). Unset it to get
# upstream's crossover back. Read only while a CustomAllreduce is actually
# serving, so it is inert when the switch above is 0.
export VLLM_CUSTOM_ALLREDUCE_ALGO=${VLLM_CUSTOM_ALLREDUCE_ALGO:-$GLM_DEFAULT_VLLM_CUSTOM_ALLREDUCE_ALGO}
# Compatibility default, independent of layout and the PCIe P2P switch.
# Preserve explicit values verbatim, including empty and composed settings.
# CustomAllreduce needs legacy CUDA IPC handles, unavailable with VMM
# (expandable_segments); the engine guards that incompatible combination.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-$GLM_DEFAULT_PYTORCH_CUDA_ALLOC_CONF}
# Report configuration, not inferred allocator precedence when both aliases exist.
printf 'serve.sh: allocator: PYTORCH_CUDA_ALLOC_CONF=%q PYTORCH_ALLOC_CONF=%q\n' \
  "$PYTORCH_CUDA_ALLOC_CONF" "${PYTORCH_ALLOC_CONF-<unset>}"
# Content-verified peer access for TP and PP, shared by both runtimes. It sets
# P2P_ENABLED and, when the check does not pass, turns every peer transport off.
source "$REPO_ROOT/p2p_check.sh"
# After a pass: the custom all-reduce owns the TP all-reduce over PCIe
# peer-to-peer (measured at TP4: decode step -5.8% at 1 user, -10.8% at 4,
# cold prefill unchanged, +21,504 KV tokens). With TP=1 there is no TP
# all-reduce and the switch stays at its engine default (off). NCCL goes over
# peer-to-peer too (NCCL_P2P_LEVEL=SYS) under TP while the custom all-reduce is
# on, and under PP for the stage send/recv: cold prefill +15.5% at TP4. An
# explicit NCCL_P2P_LEVEL always wins; GLM5_NCCL_P2P_SYS=0 leaves NCCL's own
# choice.
if [ "${P2P_ENABLED:-0}" = 1 ]; then
  if [ "$TP" -gt 1 ]; then
    export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=${VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:-$GLM_DEFAULT_VLLM_CUSTOM_ALLREDUCE_PCIE_P2P}
  fi
  if [ "${GLM5_NCCL_P2P_SYS:-1}" = 1 ] && { { [ "$TP" -gt 1 ] && [ "${VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:-}" = 1 ]; } || [ "$PP" -gt 1 ]; }; then
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-$GLM_DEFAULT_NCCL_P2P_LEVEL}
  fi
fi
echo "serve.sh: [p2p] enabled=${P2P_ENABLED:-0} custom_all_reduce=${VLLM_CUSTOM_ALLREDUCE_PCIE_P2P:-engine default} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-unset}"

# Shared-expert overlap: enqueue the routed experts first, then submit the MoE
# shared experts to the aux stream, so the two actually run at the same time.
# With the upstream order the shared experts were submitted before the gate and
# had retired before the routed Marlin kernels were queued behind them.
# Measured alongside the sparse-MLA decode retune: shared-expert GEMM time
# overlapping the routed Marlin kernels 0.01% -> 73.3% on a rank-0 decode
# trace; decode step 17.155 -> 17.010 ms at 1 user and 32.27 -> 32.10 at 4,
# against 0.03 and 0.04 ms between two unchanged restarts; cold prefill, KV
# cache size, time to first token and GSM8K unchanged, and log-probabilities
# within what two restarts of the same server show.
# Decode only: the 256-token shared-experts stream threshold keeps the
# 1152-token prefill chunks off this path entirely.
# Kill switch: set it to 0 and restart, no rebuild.
export VLLM_SHARED_EXPERT_REORDER=${VLLM_SHARED_EXPERT_REORDER:-$GLM_DEFAULT_VLLM_SHARED_EXPERT_REORDER}

# Second-generation decode kernels and drafter RoPE sizing, on in both layouts.
# Kill switch for any of the switches below: set it to 0 and restart.
#   VLLM_AMPERE_DECODE_IDX_GLUE    sparse-attention indexer decode glue folded
#                                  into fewer kernels, plus the MoE shared add
#   VLLM_MOE_DECODE_ROUTE_V2       MoE gate GEMV + top-k + align in one kernel
#                                  (up to 32 tokens)
#   VLLM_GLM5_DECODE_MHC_V2        mHC decode (up to 32 tokens)
#   VLLM_GLM5_DRAFTER_ROPE_FIT     size the DFlash2 drafter's RoPE cache to the
#                                  context instead of 1M positions: output-
#                                  identical, more KV
# The decode kernels need VLLM_AMPERE_DECODE_KERNELS=1 (set above). New
# thin-GEMM rows for 24-row batches ride VLLM_THIN_GEMM and need no switch.
# KDA decode is consolidated under VLLM_AMPERE_DECODE_KERNELS.
# Measured together, TP=4, PCIe peer-to-peer on, decode step before -> after:
#   1 user 16.26 -> 15.05 ms, 4 users 29.98 -> 27.76, 6 users 42.00 -> 34.62,
#   8 users 43.92 -> 41.55;
#   cold prefill flat; KV +2,048 tokens (RoPE fit +12,288, kernels -10,240).
TP4V2_DEFAULT=$GLM_DEFAULT_TP4V2
for _v in VLLM_AMPERE_DECODE_IDX_GLUE VLLM_MOE_DECODE_ROUTE_V2 VLLM_GLM5_DECODE_MHC_V2 VLLM_GLM5_DRAFTER_ROPE_FIT; do
  export "$_v=${!_v:-$TP4V2_DEFAULT}"
done
echo "serve.sh: second-generation decode: IDX_GLUE=$VLLM_AMPERE_DECODE_IDX_GLUE MOE_ROUTE_V2=$VLLM_MOE_DECODE_ROUTE_V2 MHC_V2=$VLLM_GLM5_DECODE_MHC_V2 DRAFTER_ROPE_FIT=$VLLM_GLM5_DRAFTER_ROPE_FIT"

# Repeatable output: a request on its own returns the same output every time: the
# same tokens and the same log-probabilities on every repeat and across
# restarts. Each fix is OFF in the engine code and turned on here, in every
# layout. Kill switch for any of them: set it to 0 and restart.
#   VLLM_MOE_DETERMINISTIC_ALIGN       MoE block alignment by counting sort, in
#                                      a fixed order (no atomics)
#   VLLM_MOE_MASK_PADDING              CUDA-graph padding rows routed to no
#                                      expert (inside the fused router when it
#                                      is on), for batches above
#                                      VLLM_MOE_MASK_PADDING_MAX_TOKENS rows
#                                      (8, the decode kernels' range)
#   VLLM_GLM5_TOPK_TIEFIX              sparse-attention indexer top-k keeps the
#                                      lowest index among exact ties
#   VLLM_GLM5_TOPK_SORTED              indexer top-k rows in ascending order
#                                      (needs VLLM_GLM5_TOPK_TIEFIX=1)
#   VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS   batches of up to this many rows run the
#                                      tie fix + sort split across more
#                                      programs (8; 0 = one program per row)
#   VLLM_FLA_PIN_AUTOTUNE              the linear-attention (KDA) prefill
#                                      kernels use one pinned configuration
#                                      per shape instead of tuning themselves
#                                      in each new process, so a fresh install
#                                      and a cleared cache give the same
#                                      output as every other start. The pinned
#                                      configurations are exactly as accurate
#                                      as the tuned ones against a 64-bit
#                                      reference, at no measurable cost.
# Measured on vs off, four alternating restarts: decode step +0.59% at 1 user,
# -1.75% at 4, -3.74% at 8 (within restart-to-restart variation), prefill
# -0.32%.
# Never set VLLM_MOE_SKIP_PADDING=0 while VLLM_MOE_MASK_PADDING=1: the
# padding mask relies on the buffer that SKIP_PADDING=1 (the engine default)
# fills, so serve.sh refuses that combination.
export VLLM_MOE_DETERMINISTIC_ALIGN=${VLLM_MOE_DETERMINISTIC_ALIGN:-$GLM_DEFAULT_VLLM_MOE_DETERMINISTIC_ALIGN}
export VLLM_MOE_MASK_PADDING=${VLLM_MOE_MASK_PADDING:-$GLM_DEFAULT_VLLM_MOE_MASK_PADDING}
export VLLM_GLM5_TOPK_TIEFIX=${VLLM_GLM5_TOPK_TIEFIX:-$GLM_DEFAULT_VLLM_GLM5_TOPK_TIEFIX}
export VLLM_GLM5_TOPK_SORTED=${VLLM_GLM5_TOPK_SORTED:-$GLM_DEFAULT_VLLM_GLM5_TOPK_SORTED}
export VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=${VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS:-$GLM_DEFAULT_VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS}
export VLLM_FLA_PIN_AUTOTUNE=${VLLM_FLA_PIN_AUTOTUNE:-$GLM_DEFAULT_VLLM_FLA_PIN_AUTOTUNE}
if [ "${VLLM_MOE_SKIP_PADDING:-1}" = "0" ] && [ "$VLLM_MOE_MASK_PADDING" = "1" ]; then
  echo "serve.sh: VLLM_MOE_SKIP_PADDING=0 with VLLM_MOE_MASK_PADDING=1 would mask real rows; refusing" >&2
  exit 2
fi
echo "serve.sh: repeatable output: MOE_ALIGN=$VLLM_MOE_DETERMINISTIC_ALIGN MASK_PADDING=$VLLM_MOE_MASK_PADDING TOPK_TIEFIX=$VLLM_GLM5_TOPK_TIEFIX TOPK_SORTED=$VLLM_GLM5_TOPK_SORTED SPLIT_ROWS=$VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS FLA_PIN=$VLLM_FLA_PIN_AUTOTUNE"

# KV headroom: return memory the engine reserves but a step cannot use to the
# KV pool. Outputs bit-identical, no measurable step-time cost. The logits
# budget applies in both layouts; the selector shard splits across
# tensor-parallel cards, so it is TP only.
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128  prefill indexer logits budget; set
#                                  512 (upstream's default) to switch it off
#   VLLM_GLM5_DRAFTER_SELECTOR_SHARD  DFlash2 selector tables split across the
#                                  cards (one extra all-reduce per draft step)
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-$GLM_DEFAULT_VLLM_SPARSE_INDEXER_MAX_LOGITS_MB}
if [ "$TP" -gt 1 ]; then
  export VLLM_GLM5_DRAFTER_SELECTOR_SHARD=${VLLM_GLM5_DRAFTER_SELECTOR_SHARD:-$GLM_DEFAULT_VLLM_GLM5_DRAFTER_SELECTOR_SHARD}
fi
echo "serve.sh: KV headroom: MAX_LOGITS_MB=$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB SELECTOR_SHARD=${VLLM_GLM5_DRAFTER_SELECTOR_SHARD:-engine default}"
# Raw-bit indexer K dequant, both layouts: the sparse-attention indexer's decode
# logits dequantize the e4m3 K tiles by bit placement on sm_80 instead of
# through fp32 (bitwise-identical logits). Off in the engine code, set here.
# Banner "DSA indexer decode logits: raw-bit e4m3 K dequant active". Kill
# switch: 0.
export VLLM_INDEXER_DECODE_RAW_K=${VLLM_INDEXER_DECODE_RAW_K:-$GLM_DEFAULT_VLLM_INDEXER_DECODE_RAW_K}
# tool_choice="none", both layouts: the glm47 tool parser makes the tool-call
# start token a banned token for the model and the drafter, so a request that
# forbids tools cannot start one. Off in the engine code, set here. Banner
# "GLM tool_choice=none decode mask: enabled". Kill switch: 0.
export VLLM_TOOL_CHOICE_NONE_MASK=${VLLM_TOOL_CHOICE_NONE_MASK:-$GLM_DEFAULT_VLLM_TOOL_CHOICE_NONE_MASK}
echo "serve.sh: indexer raw-bit K=$VLLM_INDEXER_DECODE_RAW_K tool_choice=none mask=$VLLM_TOOL_CHOICE_NONE_MASK"

# Prefill chunk: batched-token budget, which sets the prefill chunk.
# The KDA state page forces chunk ends onto 1152-token blocks, and DFlash
# reserves its draft slots out of the budget, so the chunk is
# floor((MAX_BATCHED - slots) / 1152) x 1152:
#   2048 -> 1152 (1.2.0), 2312 -> 2304, 3463 -> 3456 (the TP4 default here).
# Larger chunks let each pass over the experts carry more tokens: 3,456-token
# chunks measured +2.4% cold prefill against 2,304 at 180 W per card, for about
# 35,600 fewer KV tokens (-3%, a larger chunk raises the activation peak the
# memory profiler reserves for). Decode step unchanged.
# Contended prefill is unaffected (FAIR_PREFILL caps it at 384 while anything
# decodes). The TP4 default follows the draft slots (the deepest adaptive depth,
# 7, or SPEC_N for mtp) to keep 3,456-token chunks: 3456 + max(4, slots). Under
# PP4 the default is 2312 (2,304-token chunks), the chunk it was validated with.
# Kill switch: MAX_BATCHED=2048 restores 1.2.0's chunking.
# Load-adaptive DFlash depth (both layouts, dflash only), through vLLM's
# --speculative-config adaptive_k mapping, one per layout: num_speculative_tokens
# 7 is the deepest verified depth; by_load [7,5,3] verifies up to 7 at one
# request, 5 at two and 3 under load; accept.costs / costs_multi are the measured
# per-depth step costs (one request / several) that let each request's
# acceptance lower its depth within those limits; under TP4 draft_by_load makes
# the drafter's width follow the verified depth. The engine reserves the deepest
# depth as draft slots. Banners: "load-adaptive DFlash depth active",
# "acceptance-aware DFlash depth active", "load-following DFlash draft width
# active" (TP4). Kill switch: pass your own --speculative-config in EXTRA_ARGS,
# for example the fixed depth 3 of earlier releases:
#   EXTRA_ARGS='--speculative-config {"method":"dflash","model":"<drafter dir>","num_speculative_tokens":3}'
# The drafter's boundary cache (VLLM_DFLASH_BOUNDARY_CACHE) is on in both
# layouts. Kill switch: 0.
export VLLM_DFLASH_BOUNDARY_CACHE=${VLLM_DFLASH_BOUNDARY_CACHE:-$GLM_DEFAULT_VLLM_DFLASH_BOUNDARY_CACHE}
OWN_SPEC=0
if [[ "${EXTRA_ARGS:-}" == *--speculative-config* ]]; then OWN_SPEC=1; fi
if [ "$MODE" = dflash ] && [ "$OWN_SPEC" = 0 ] && [ -n "${SPEC_N:-}" ]; then
  echo "serve.sh: SPEC_N applies to mtp only; dflash depths come from the layout's adaptive_k mapping (unset SPEC_N, or pass your own --speculative-config in EXTRA_ARGS for another depth)" >&2
  exit 2
fi
if [ "$PP" -gt 1 ]; then ADAPTIVE_K_JSON=$GLM_DEFAULT_ADAPTIVE_K_PP4; else ADAPTIVE_K_JSON=$GLM_DEFAULT_ADAPTIVE_K_TP4; fi
if [ "$MODE" = dflash ] && [ "$OWN_SPEC" = 0 ]; then
  SPEC_DEPTH=$GLM_DEFAULT_DFLASH_DEPTH
  echo "serve.sh: [adaptive-depth] adaptive_k=$ADAPTIVE_K_JSON draft slots=$SPEC_DEPTH boundary=$VLLM_DFLASH_BOUNDARY_CACHE"
else
  SPEC_DEPTH=${SPEC_N:-$GLM_DEFAULT_SPEC_N}
  echo "serve.sh: [adaptive-depth] adaptive_k=$([ "$OWN_SPEC" = 1 ] && echo 'from EXTRA_ARGS' || echo off) draft slots=$SPEC_DEPTH"
fi
# TP4 keeps 3,456-token chunks for any draft depth: 3456 + max(4, depth).
if [ "$TP" -gt 1 ]; then
  MAX_BATCHED_DEFAULT=$((GLM_DEFAULT_CHUNK_TP4 + (SPEC_DEPTH > GLM_DEFAULT_MIN_DRAFT_SLOTS ? SPEC_DEPTH : GLM_DEFAULT_MIN_DRAFT_SLOTS)))
else
  MAX_BATCHED_DEFAULT=$GLM_DEFAULT_MAX_BATCHED_PP4
fi
# Fair prefill: decode-aware chunking, passed as real serve args below rather
# than through EXTRA_ARGS so EXTRA_ARGS stays free for callers.
FAIR_ARGS=()
if [ "${FAIR_PREFILL:-$GLM_DEFAULT_FAIR_PREFILL}" = "1" ]; then
  FAIR_ARGS=(--prefill-chunk-with-decodes "${FAIR_CHUNK:-$GLM_DEFAULT_FAIR_CHUNK}" --max-num-partial-prefills "${FAIR_PARTIAL:-$GLM_DEFAULT_FAIR_PARTIAL}")
fi

# Multimodal: vision + video stay on. Default is UNCAPPED (the profiler reserves for a
# context-filling video on stage 0, ~150k fewer KV tokens). Set MM_CAP=1 to bound inputs instead:
# profiler dummy = MM_IMAGES images + 1 video of MM_FRAMES frames, and the same bound is enforced
# on real inputs (media-io-kwargs frames, mm-processor-kwargs tokens).
# The image and video processors and the engine's profiler all read max_image_tokens (one token per
# 28x28 = 784 px of image area; for video the cap is per whole video), so it alone is passed: real
# items and the profiled items, which size the encoder cache, then agree (recipe issue #11).
# MM_MAX_IMAGE_TOKENS defaults to MM_MAX_PIXELS / 784 (1003520 px = 1280 tokens).
MM_IMAGES=${MM_IMAGES:-$GLM_DEFAULT_MM_IMAGES}; MM_FRAMES=${MM_FRAMES:-$GLM_DEFAULT_MM_FRAMES}; MM_MAX_PIXELS=${MM_MAX_PIXELS:-$GLM_DEFAULT_MM_MAX_PIXELS}
MM_MAX_IMAGE_TOKENS=${MM_MAX_IMAGE_TOKENS:-$(( MM_MAX_PIXELS / 784 ))}
MM_ARGS=()
if [ "${MM_CAP:-$GLM_DEFAULT_MM_CAP}" = "1" ]; then MM_ARGS=(
  --limit-mm-per-prompt "{\"image\":$MM_IMAGES,\"video\":{\"count\":1,\"num_frames\":$MM_FRAMES,\"width\":$GLM_DEFAULT_MM_WIDTH,\"height\":$GLM_DEFAULT_MM_HEIGHT}}"
  --mm-processor-kwargs "{\"max_image_tokens\":$MM_MAX_IMAGE_TOKENS}"
  --media-io-kwargs "{\"video\":{\"num_frames\":$MM_FRAMES}}"
); fi

SPEC=()
if [ "$OWN_SPEC" = 1 ]; then
  : # the caller supplied its own speculative-config; never add a second one
elif [ "$MODE" = "mtp" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_N:-$GLM_DEFAULT_SPEC_N}}")
elif [ "$MODE" = "dflash" ]; then
  # The layout's validated adaptive-depth configuration (see above).
  SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":$GLM_DEFAULT_DFLASH_DEPTH,\"adaptive_k\":$ADAPTIVE_K_JSON}")
fi

# Empty keeps the model's template default. Explicit CLI defaults win.
REASONING_ARGS=()
if [ -z "${DEFAULT_REASONING_EFFORT:-}" ]; then
  echo "serve.sh: [reasoning-effort] template default"
elif [[ "${EXTRA_ARGS:-}" == *--default-chat-template-kwargs* ]]; then
  echo "serve.sh: [reasoning-effort] explicit EXTRA_ARGS defaults"
else
  case "$DEFAULT_REASONING_EFFORT" in
    low|high|max) ;;
    *) echo "[reasoning-effort] DEFAULT_REASONING_EFFORT must be empty, low, high or max" >&2; exit 2 ;;
  esac
  REASONING_ARGS=(--default-chat-template-kwargs "{\"reasoning_effort\":\"$DEFAULT_REASONING_EFFORT\"}")
  echo "serve.sh: [reasoning-effort] default=$DEFAULT_REASONING_EFFORT (request overrides win)"
fi

# The key goes to vLLM through the environment rather than --api-key, so it
# does not show up in the process list.
if [ -n "${API_KEY:-}" ]; then export VLLM_API_KEY="$API_KEY"; fi

CMD=("$VENV/bin/vllm" serve "$MODEL"
  --served-model-name "${SERVED_MODEL_NAME:-${SERVED_NAME:-$GLM_DEFAULT_SERVED_MODEL_NAME}}"
  --host "${HOST:-$GLM_DEFAULT_HOST}"
  --pipeline-parallel-size "$PP"
  --tensor-parallel-size "$TP"
  --max-model-len "${MAX_LEN:-$GLM_DEFAULT_MAX_LEN}"
  --max-num-seqs "${MAX_SEQS:-$GLM_DEFAULT_MAX_SEQS}"
  --max-num-batched-tokens "${MAX_BATCHED:-$MAX_BATCHED_DEFAULT}"
  --long-prefill-token-threshold "${PREFILL_CAP:-$GLM_DEFAULT_PREFILL_CAP}"
  --gpu-memory-utilization "${GPU_UTIL:-$GLM_DEFAULT_GPU_UTIL}"
  "$GLM_DEFAULT_TRUST_REMOTE_CODE"
  --reasoning-parser "${REASONING_PARSER:-$GLM_DEFAULT_REASONING_PARSER}"
  "$GLM_DEFAULT_AUTO_TOOL_CHOICE" --tool-call-parser "${TOOL_PARSER:-$GLM_DEFAULT_TOOL_PARSER}"
  --port "${PORT:-$GLM_DEFAULT_PORT}"
  ${MM_ARGS[@]+"${MM_ARGS[@]}"} ${SPEC[@]+"${SPEC[@]}"} ${SAMPLING_ARGS[@]+"${SAMPLING_ARGS[@]}"} ${FAIR_ARGS[@]+"${FAIR_ARGS[@]}"}
  ${BLOCK_ARGS[@]+"${BLOCK_ARGS[@]}"} ${REASONING_ARGS[@]+"${REASONING_ARGS[@]}"} ${EXTRA_ARGS:-})

BOOT_CHECK=${BOOT_CHECK:-$GLM_DEFAULT_BOOT_CHECK}
case "$BOOT_CHECK" in 0|1) ;; *) echo "[boot-check] FAIL: BOOT_CHECK must be 0 or 1" >&2; exit 2 ;; esac
if [ "${DRY:-0}" = "1" ]; then
  echo "[boot-check] BOOT_CHECK=$BOOT_CHECK (dry run)"
  echo "serve.sh: DRY=1, layout $LAYOUT (PP=$PP TP=$TP), environment the server would get:"
  env | LC_ALL=C sort | grep -E '^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=)' | grep -v '^VLLM_API_KEY=' | sed 's/^/  /' || true
  if [ -n "${VLLM_API_KEY:-}" ]; then echo "  VLLM_API_KEY=(set, not shown)"; else echo "  (no API key: /v1 is open to anyone who can reach ${HOST:-$GLM_DEFAULT_HOST}:${PORT:-$GLM_DEFAULT_PORT})"; fi
  echo "serve.sh: DRY=1, command:"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
if [ "$BOOT_CHECK" = 0 ]; then
  echo "[boot-check] disabled (BOOT_CHECK=0)"
elif [ "${RECIPE_BOOT_CHECK_OWNER:-}" = start ]; then
  echo "[boot-check] delegated to start.sh after health"
else
  CLIENT_HOST=${HOST:-$GLM_DEFAULT_HOST}
  case "$CLIENT_HOST" in 0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;; esac
  boot_args=()
  [ -z "${RECIPE_CACHE_ROOT:-}" ] || boot_args+=(--container)
  exec "$VENV/bin/python" "$REPO_ROOT/boot_check.py" "${boot_args[@]}" \
    --base "http://$CLIENT_HOST:${PORT:-$GLM_DEFAULT_PORT}" \
    --model "${SERVED_MODEL_NAME:-${SERVED_NAME:-$GLM_DEFAULT_SERVED_MODEL_NAME}}" \
    --ready-timeout "${READY_TIMEOUT:-$GLM_DEFAULT_READY_TIMEOUT}" --serve "${CMD[@]}"
fi
exec "${CMD[@]}"
