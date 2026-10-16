# Engine switches

The engine features behind the published numbers are controlled by about
forty internal variables (`VLLM_AMPERE_*`, `VLLM_GLM5_*`, `VLLM_PP_*`,
`VLLM_MOE_*`, `VLLM_KV_*`, `VLLM_CUSTOM_ALLREDUCE_*`, `PYTORCH_CUDA_ALLOC_CONF`
and a few others) and two vLLM options (`--speculative-config` and
`--enable-batch-sharded-sampling`). You do not need any of them to run the
model: `serve.sh` sets every one to its release default. They exist so that
each feature can be switched off on its own, which makes them the first tool
for troubleshooting.

Each is listed, with its default, in
[`.env.advanced.example`](../.env.advanced.example). To change one, put the line
in your `.env` (`./start.sh` reads only `.env`), or set it for one run on the
command line. For a container started by hand or with Docker Compose, set it in
[`docker/container.env`](../docker/container.env). Settings renamed in this
release are listed under [Renamed and retired settings](#renamed-and-retired-settings).

## Troubleshooting with kill switches

If output quality is ever in question, try `VLLM_AMPERE_DECODE_KERNELS=0` first.
If a problem goes away with one feature switched off, that feature is the
place to look; include the switch and the result when you report it.


Each feature is one variable. Set it to `0` and restart; no rebuild, no
revert. The PP4 draft tail is switched off with
`VLLM_PP_DRAFT_TAIL_STAGE=-1` (default `2`), `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
with `512`, and batch-sharded sampling with
`EXTRA_ARGS=--no-enable-batch-sharded-sampling`.

```bash
VLLM_AMPERE_DECODE_KERNELS=0 ./start.sh restart
```

| Variable | Feature | Layout |
|---|---|---|
| `VLLM_PP_DRAFT_TAIL_STAGE` | stage `2` runs the drafter's final step; `-1` switches it off | PP4 |
| `VLLM_GLM5_PREFILL_KERNELS` | Ampere prefill kernels (mHC projection, sparse attention) | both |
| `VLLM_GLM5_SMLA_PREFILL_PRED_LOAD` | sparse-attention prefill gather that skips empty slots | both |
| `VLLM_AMPERE_KDA_PREFILL_NARROW`, `VLLM_AMPERE_MARLIN_PREFILL_NARROW` | linear-attention and MoE prefill kernels at TP4 shapes | TP4 |
| `VLLM_AMPERE_KDA_PREFILL_WIDE`, `VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE`, `VLLM_AMPERE_MARLIN_PREFILL_WIDE` | linear-attention, sparse-attention and MoE prefill kernels at PP4 shapes | PP4 |
| `VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED` | compiled wide-tile tiles for the PP4 MoE prefill | PP4 |
| `VLLM_PP_SPREAD_DECODES` | decodes spread over every in-flight micro-batch | PP4 |
| `VLLM_PP_PACKED_HOP`, `VLLM_PP_HOP_NO_METADATA` | packed, metadata-free hand-off between stages | PP4 |
| `VLLM_PP_SPLIT_DRAFT_EVENT`, `VLLM_GLM5_PP_FOLD_DRAFT_FC` | drafter synchronisation and input projection | PP4 |
| `--enable-batch-sharded-sampling` (vLLM option) | batch-sharded logits and sampling; `EXTRA_ARGS=--no-enable-batch-sharded-sampling` switches it off | TP4 |
| `VLLM_AMPERE_DECODE_KERNELS` | fused decode kernels | both |
| `VLLM_AMPERE_DECODE_IDX_GLUE`, `VLLM_MOE_DECODE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2` | second-generation decode kernels | both |
| `VLLM_GLM5_DRAFTER_ROPE_FIT` | drafter position table sized to the context (more KV) | both |
| `VLLM_AMPERE_KDA_FOLD`, `VLLM_THIN_GEMM_CONCAT`, `VLLM_AMPERE_KDA_SMALL_STACK` | fewer decode launches: KDA state commit folded into the next verify, attention thin-GEMM groups as one launch, small-stack KDA verify configuration | TP4 |
| `VLLM_AMPERE_MARLIN_DECODE_V2` | compiled Marlin decode v2 (needs `VLLM_AMPERE_MARLIN_DECODE_CUDA=1`) | TP4 |
| `VLLM_THIN_GEMM` | small-batch GEMMs | both |
| `VLLM_HOST_ALLREDUCE` | host-staged all-reduce | TP4 |
| `VLLM_SHARED_EXPERT_REORDER` | shared experts overlapped with the routed experts | both |
| `FAIR_PREFILL` | fair prefill | both |
| `VLLM_MOE_DETERMINISTIC_ALIGN` | same output every time: MoE block alignment in a fixed order | both |
| `VLLM_MOE_MASK_PADDING` | same output every time: CUDA-graph padding rows kept out of the MoE, for batches above `VLLM_MOE_MASK_PADDING_MAX_TOKENS` rows (8) | both |
| `VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED` | same output every time: indexer top-k consistent on ties, in a fixed order (`TOPK_SORTED=1` needs `TOPK_TIEFIX=1`) | both |
| `VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS` | the two above spread over more programs for batches up to `8` rows; `0` = one per row | both |
| `VLLM_FLA_PIN_AUTOTUNE` | same output on every install: pinned linear-attention prefill configurations | both |
| `VLLM_KV_MAMBA_INFLIGHT_STATES` | KV reserve for the linear-attention states of prefill chunks in flight; `0` reports the old, larger pool | both |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | KV headroom: prefill indexer logits budget, `128` here; `512` (upstream's) switches it off | both |
| `VLLM_GLM5_DRAFTER_SELECTOR_SHARD` | KV headroom: drafter selector tables split across the cards | TP4 |
| `VLLM_INDEXER_DECODE_RAW_K` | indexer decode K dequantized by bit placement instead of through fp32, bitwise-identical | both |
| `VLLM_TOOL_CHOICE_NONE_MASK` | with `tool_choice="none"`, the model and the drafter cannot start a tool call | both |
| `VLLM_DFLASH_BOUNDARY_CACHE` | cached prompt-boundary reuse in the DFlash2 drafter | both |

`DRY=1 ./start.sh` prints the container command and the environment the server
would get, so you can check what is on. It runs the preflight and says whether
a real start would pull or download, but pulls, downloads and launches nothing,
so it also works before the first install.

CPU regression checks live in `tools/tests`: run `python -m pytest -q tools/tests`.
They exercise slow frontend readiness and the launcher's depth and kill-switch
settings. With the pinned fork's source importable, they also parse the
launcher's adaptive-depth configuration with the engine's own speculative-config
validation, without loading a model or initialising CUDA; those engine-contract
cases are skipped otherwise. `RECIPE_TEST_ENGINE=/path/to/fork` selects a
specific source tree.

## Draft depth

`serve.sh` passes the layout's measured adaptive-depth configuration to vLLM as
`--speculative-config` with `num_speculative_tokens` 7 and an `adaptive_k`
mapping: `by_load` `[7,5,3]` verifies up to 7 draft tokens at one request, 5 at
two and 3 under load; `accept.costs` / `accept.costs_multi` are the per-depth
step costs measured on each layout (one request / several), which let each
request's acceptance lower its depth within those limits; under TP4
`draft_by_load` makes the drafter compute only the verified depth. The TP4 token
budget follows the deepest depth (3456 + 7 = 3463). `SPEC_N` applies to `mtp`
only; to run DFlash2 at another depth, pass your own configuration, which
replaces this one, for example the fixed depth 3 of earlier releases:

```bash
EXTRA_ARGS='--speculative-config {"method":"dflash","model":"/models/GLM-5.3-Flash-DFlash2","num_speculative_tokens":3}'
```

## Other engine settings

| Variable | Default | What it does |
|---|---|---|
| `VLLM_GLM5_PREFILL_MIN_TOKENS` | `384` | chunk-size gate for the Ampere prefill kernels; 384, not the code default 512, because `FAIR_CHUNK` caps the chunk at 384 while something decodes |
| `VLLM_AMPERE_MARLIN_DECODE_CUDA` | on if installed | optional compiled Marlin decode (split-K reduction order); see [Optional compiled Marlin](compiled-marlin.md) |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS` | `1` at TP4 | flags-in-data all-reduce with verified P2P; `0` disables it |
| `VLLM_PP_LAYER_PARTITION` | `13,11,11,10` | PP4 layer split |
| `VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS`, `VLLM_GLM5_MARLIN_DECODE_PP_MULTI` | `1` at PP4; engine default `0` elsewhere | eligible compiled PP4 decode at 5–7 rows and multi-request rows; optional library required |
| `VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS` | `9-64` at PP4; engine default `9-32` elsewhere | compiled PP4 multi-request row range |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES`, `VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S` | `262144`, `60` (engine defaults) | flags-in-data byte gate and peer wait limit in seconds; an overrun traps |
| `VLLM_STATE_INDEX_CHECK` | `0` (engine default) | host state-index diagnostic, for validation runs |
| `VLLM_GLM5_DECODE_KDA_WIDE` | `1` (engine default) | consolidated KDA decode for measured 64-head PP4 shapes; requires `VLLM_AMPERE_DECODE_KERNELS=1`; TP4 is unaffected |
| `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` | `1` under TP4 after a content-check pass | the custom all-reduce over PCIe peer-to-peer; `0` keeps the host-staged all-reduce. Inert under PP4 |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | `2stage` | which CustomAllreduce kernel; the built-in crossover is NVLink-tuned. Inert unless [peer-to-peer](how-to-use.md#pcie-peer-to-peer) is on |
| `GLM5_NCCL_P2P_SYS` | `1` | after the content check passes, sets `NCCL_P2P_LEVEL=SYS` under TP while the custom all-reduce is on, and under PP; `0` suppresses this automatic setting |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | allocator compatibility default, independent of layout and peer-to-peer; explicit values are preserved |
| `aux_hidden_state_tensor` (model configuration field) | the model's own | which hidden-state tensor the DFlash2 drafter reads; set it with `--hf-overrides '{"text_config": {"aux_hidden_state_tensor": "branch"}}'` in `EXTRA_ARGS` |

## Renamed and retired settings

These names are retired in this release. The engine ignores them and warns once
at startup when one is set (`... is retired and ignored: ...`); the warning goes
away in a later release. Remove them from `.env`, or use the replacement.

| Retired | Use instead |
|---|---|
| `VLLM_GLM5_DECODE_KERNELS` | `VLLM_AMPERE_DECODE_KERNELS` |
| `VLLM_GLM5_DECODE_IDX_GLUE`, `VLLM_GLM5_DECODE_MOE_ROUTE_V2` | `VLLM_AMPERE_DECODE_IDX_GLUE`, `VLLM_MOE_DECODE_ROUTE_V2` |
| `VLLM_GLM5_DETERMINISTIC_MOE_ALIGN` | `VLLM_MOE_DETERMINISTIC_ALIGN` |
| `VLLM_GLM5_MOE_MASK_PADDING` | `VLLM_MOE_MASK_PADDING`; batches of up to `VLLM_MOE_MASK_PADDING_MAX_TOKENS` rows (8) are left unmasked even with the decode kernels off (set it to `0` to mask every batch) |
| `VLLM_GLM5_DECODE_MOE_MAX_TOKENS` | `VLLM_MOE_MASK_PADDING_MAX_TOKENS` |
| `VLLM_GLM5_THIN_GEMM`, `VLLM_GLM5_THIN_GEMM_MAX_TOKENS` | `VLLM_THIN_GEMM`, `VLLM_THIN_GEMM_MAX_TOKENS` |
| `VLLM_GLM5_HOST_ALLREDUCE`, `VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE` | `VLLM_HOST_ALLREDUCE`, `VLLM_HOST_ALLREDUCE_MAX_SIZE` |
| `VLLM_GLM5_SHARED_EXPERT_REORDER` | `VLLM_SHARED_EXPERT_REORDER` |
| `VLLM_GLM5_FLA_PIN_AUTOTUNE` | `VLLM_FLA_PIN_AUTOTUNE` |
| `VLLM_GLM5_MEM_ATTRIBUTION`, `VLLM_GLM5_STATE_INDEX_CHECK` | `VLLM_MEM_ATTRIBUTION`, `VLLM_STATE_INDEX_CHECK` |
| `VLLM_GLM5_INDEXER_DECODE_RAW_K`, `VLLM_GLM5_TOOL_CHOICE_NONE_MASK` | `VLLM_INDEXER_DECODE_RAW_K`, `VLLM_TOOL_CHOICE_NONE_MASK` |
| `VLLM_DSA_INDEXER_REF` | `VLLM_INDEXER_DSA_REF` |
| `VLLM_GLM5_DFLASH_BOUNDARY_CACHE` | `VLLM_DFLASH_BOUNDARY_CACHE` |
| `VLLM_GLM5_MARLIN_DECODE_CUDA` | `VLLM_AMPERE_MARLIN_DECODE_CUDA` |
| `VLLM_BUILD_AMPERE_MARLIN` | `VLLM_AMPERE_MARLIN_BUILD` (engine builds only; the image already includes the library) |
| `VLLM_GLM5_TP4_KDA_PREFILL`, `VLLM_GLM5_TP4_MARLIN_PREFILL` | `VLLM_AMPERE_KDA_PREFILL_NARROW`, `VLLM_AMPERE_MARLIN_PREFILL_NARROW` |
| `VLLM_GLM5_PP_KDA_PREFILL`, `VLLM_GLM5_PP_SPARSE_MLA_PREFILL`, `VLLM_GLM5_PP_MARLIN_PREFILL`, `VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED` | `VLLM_AMPERE_KDA_PREFILL_WIDE`, `VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE`, `VLLM_AMPERE_MARLIN_PREFILL_WIDE`, `VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED` |
| `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE` | `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K`, `_K_DEPTHS`, `_ADAPTIVE_DRAFT_WIDTH`, `_K_ACCEPT`, `_K_COSTS`, `_K_COSTS_MULTI`, `_K_HYST`, `_K_PRIOR`, `_K_LOG` (all `VLLM_GLM5_DFLASH_…`) | `--speculative-config` `adaptive_k`: `by_load`, `draft_by_load`, `accept.costs`, `accept.costs_multi`, `accept.hysteresis`, `accept.prior`, `log_interval` ([Draft depth](#draft-depth)) |
| `VLLM_GLM5_LOCAL_LOGITS` | `--enable-batch-sharded-sampling` (`EXTRA_ARGS=--no-enable-batch-sharded-sampling` switches it off) |
| `VLLM_GLM5_AUX_HIDDEN_TENSOR` | model configuration field `aux_hidden_state_tensor` (`--hf-overrides`) |
| `VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR` | `TORCH_EXTENSIONS_DIR` |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR` | `$VLLM_CACHE_ROOT/custom_all_reduce_flags` |
| `VLLM_GLM5_PREFILL_OVERLAP` and its `_SPLITS`, `_MIN_TOKENS`, `_BACKEND`, `_DEBUG`, `_UNDER_CA` settings, `VLLM_GLM5_PREFILL_OBSERVE` | removed; the TP all-reduces of a prefill run unsplit, as in vLLM |
| `VLLM_GLM5_DECODE_MHC`, `VLLM_GLM5_DECODE_MHC_MAX_TOKENS`, `VLLM_GLM5_DECODE_MHC_V3` | removed; `VLLM_GLM5_DECODE_MHC_V2` switches the one remaining mHC decode kernel |
| `VLLM_GLM5_DECODE_MHC_V2_FN_BF16` | removed; mHC decode reads the fp32 weights |
| `VLLM_GLM5_THIN_GEMM_V74`, `THIN_GEMM_MAX_M` | removed; one thin-GEMM schedule, bounded by `VLLM_THIN_GEMM_MAX_TOKENS` |
| `VLLM_GLM5_PREFILL_PACK_BF16X2` | removed; one mHC pre-norm pack |
| `VLLM_GLM5_MARLIN_DECODE_VARIANT` | removed; compiled Marlin decode uses the split-K order (the former default, `orig`) |
| `VLLM_GLM5_CUSTOM_ALLREDUCE_MAX_SIZE` | removed; the custom all-reduce uses its 8 MiB default cap |
| `VLLM_GLM5_INDEXER_DECODE_ROWS`, `VLLM_GLM5_INDEXER_GATHER_CLAMP` | removed; the indexer decode tables and gather workspace are sized as in vLLM |
| `VLLM_KV_SWA_INFLIGHT_SCRATCH` | removed; the KV report charges sliding-window blocks in flight per concurrency slot, as in vLLM |
| `VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED` | removed; the TP4 MoE prefill runs on the released Marlin kernels |

The `VLLM_GLM5_PROLOGUE_FUSE`, `VLLM_GLM5_DECODE_KDA_V2`,
`VLLM_GLM5_DECODE_KDA_V2_DEEP`, `VLLM_GLM5_DECODE_KDA_STEP_TILE`,
`VLLM_GLM5_KDA_RECOVER`, `VLLM_GLM5_MARLIN_PREFILL_CUDA`,
`VLLM_GLM5_REPLICATED_EMBED` and `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY` settings of
earlier releases are no longer read by this engine, without a warning; KDA
decode is consolidated under `VLLM_AMPERE_DECODE_KERNELS`.
