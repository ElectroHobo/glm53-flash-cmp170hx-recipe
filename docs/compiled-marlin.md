# Optional compiled Marlin

The launcher turns compiled decode on in both layouts when the optional
library is installed. The effective `TP=4 PP=1` or `PP=4 TP=1` configuration selects the default even when
those dimensions override `LAYOUT`. One `vllm._ampere_marlin_C` library serves
both layouts; changing layout never rebuilds it. The decode switch
overrides the default explicitly:

```bash
VLLM_AMPERE_MARLIN_DECODE_CUDA=0 ./start.sh restart
```

Compiled decode uses one reduction order, which splits the first MoE
projection four ways along K (the default `orig` order of earlier releases;
the `VLLM_GLM5_MARLIN_DECODE_VARIANT` switch is retired). Because it adds in a
different order from the released kernels, the decoded text can differ from
them within their accuracy bounds. Under TP4 the decode v2 kernel
(`VLLM_AMPERE_MARLIN_DECODE_V2`, default on with the compiled decode) replaces
the first compiled decode kernel; PP4 keeps the first kernel:

```bash
VLLM_AMPERE_MARLIN_DECODE_V2=0 ./start.sh restart
```

An explicit `0` or `1` in the environment or `.env` remains authoritative across
updates. Leave the flag commented out to follow the defaults. Unset decode
uses CPU-only module discovery, without importing the extension or probing CUDA.
If the library is absent it defaults off with a banner; if present it defaults on
and startup must validate compatibility. Explicit `1` with an absent or incompatible
library, or default-on with an incompatible library, fails before serving, never
silently disabling the requested feature. Decode off means no extension load.
`DRY=1` prints the resolved settings and command but skips compatibility validation.

The compiled decode path is limited to eligible small batches in TP4 and PP4.
Unsupported shapes use the released paths. This is not a universal kernel
replacement, and no whole-server speedup is claimed here.

The release includes PP4 compiled wide-tile prefill
(`VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED`, on under PP4); the TP4 MoE prefill
runs on the released Marlin kernels.
Shapes outside their gates use the existing split paths. In 1.7.2, prefill
work blocks start in dependency order to avoid stalls; use the matching
engine pin and rebuilt library.
The old `VLLM_GLM5_MARLIN_PREFILL_CUDA` switch is ignored. The compiled library
is rebuilt from `819146023facb31c04f1f8512e52464835a10d40`; decode entry points retain their established outputs.
The published [Results](results.md) remain the 1.7.0 measurements until
1.7.2 validation is complete.

The pinned container image includes the optional library. Enabling the runtime
switch without the library fails before serving, never silently falling back.
Decode off means no optional-extension load.

The [container build](../docker/README.md#build-it) prebuilds the same library with
no visible GPUs. Marlin needs no startup compiler or layout-specific rebuild;
other engine kernels still need the image's existing toolkit.
