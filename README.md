# vLLM for Apertus 1.5

GPU-native Apertus 1.5 multimodal serving for vLLM.

**This is a temporary build repository.** It patches upstream vLLM with
PR #50496 (GPU-native Apertus multimodal encoders) and vendored tokenizer
files to replace the unmerged transformers PR #47662 dependency. Once both
PRs are merged into their respective upstream releases, this repo will be
archived (the Docker image will still work, but users should switch to
the upstream vLLM image instead.

> Thanks to [Swiss AI](https://huggingface.co/swiss-ai), [blancsw](https://github.com/blancsw),
  [Anunay-Yadav](https://github.com/Anunay-Yadav), [Oleg](https://github.com/loleg),
  [Cyrilvallez](https://github.com/Cyrilvallez), [AryanAhadinia](https://github.com/AryanAhadinia),
  [robmsmt](https://github.com/robmsmt), and [Neural Magic](https://github.com/neuralmagic) -
  see [Attribution](#attribution) for full credits.

## Upstream Dependencies

This image builds on two unmerged upstream PRs:

| PR | Repository | Purpose | Status |
|----|------------|---------|--------|
| [#50496](https://github.com/vllm-project/vllm/pull/50496) | vllm-project/vllm | GPU-native `Apertus1p5ForConditionalGeneration` multimodal encoders (vision: Emu3.5, audio: WavTokenizer) + native apertus tool call format + apertus reasoning parser | Open |
| [#47662](https://github.com/huggingface/transformers/pull/47662) | huggingface/transformers | `Apertus1p5VisionTokenizerModel` image/audio preprocessor for the HuggingFace transformers library | Open |

### What happens when PR #50496 is merged

- Our vendored `apertus_mm.py` and the registry patch (`0001-registry.diff`)
  ship upstream. This repo can remove them and point to the upstream release tag.
- The reasoning parser (`0002-reasoning-init.diff` + `apertus_reasoning_parser.py`)
  also ships upstream.

### What happens when PR #47662 is merged (transformers release)

- Our vendored `apertus_emu35.py`, `apertus_wavetokenizer.py`, and `apertus_utils.py`
  are no longer needed. The model code can use the upstream transformers tokenizers.
- Once both PRs land and a vLLM release includes them, this repo is archived.

## Quick Start

... (build content) ...

On a 62 GiB no-swap host, `MAX_JOBS=4` uses approximately 15-24 GiB peak RAM
and completes in roughly 20-25 minutes. Do not remove or increase these limits
without measuring peak host RAM usage. An unbounded parallel build (`-j$(nproc)`)
can make a no-swap host unresponsive.

## File Structure

| File | Purpose |
|------|---------|
| `Dockerfile` | Build recipe (clones upstream vLLM main, applies PR #50496 patches) |
| `vendored/apertus_mm.py` | Adapted from PR #50496, uses vendored tokenizer loading |
| `vendored/apertus_emu35.py` | Vendored Emu3.5 vision encoder (from Infomaniak/vllm) |
| `vendored/apertus_wavetokenizer.py` | Vendored WavTokenizer audio encoder (from Infomaniak/vllm) |
| `vendored/apertus_utils.py` | Tokenizer helper wrappers for on-disk weight loading |
| `patches/0001-registry.diff` | Adds `Apertus1p5ForConditionalGeneration` to vLLM model registry |
| `patches/0002-reasoning-init.diff` | Registers apertus reasoning parser |
| `patches/apertus_reasoning_parser.py` | Apertus reasoning parser (PR #50496) |

---

## Supported Models

- [`onprem-ai/Apertus-v1.5-70B-NVFP4`](https://huggingface.co/onprem-ai/Apertus-v1.5-70B-NVFP4) (NVFP4 three-tier mixed-precision, 48 GiB)
- [`onprem-ai/Apertus-v1.5-70B-FP8`](https://huggingface.co/onprem-ai/Apertus-v1.5-70B-FP8) (FP8 float-quantized, 71 GiB)
- [`onprem-ai/Apertus-v1.5-8B-NVFP4`](https://huggingface.co/onprem-ai/Apertus-v1.5-8B-NVFP4) (NVFP4, smaller footprint)
- [`onprem-ai/Apertus-v1.5-8B-FP8`](https://huggingface.co/onprem-ai/Apertus-v1.5-8B-FP8) (FP8 float-quantized)
- [`swiss-ai/Apertus-v1.5-70B`](https://huggingface.co/swiss-ai/Apertus-v1.5-70B) (BF16, official release)
- [`swiss-ai/Apertus-v1.5-8B`](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) (BF16, official release)
- Same architecture (`Apertus1p5ForConditionalGeneration`) in any
  compressed-tensors format (FP8_DYNAMIC, NVFP4, etc.)
- `ApertusForCausalLM` (backward-compatible text-only architecture)

---

## Quantization Support

Any checkpoint using the `compressed-tensors` format works:
- BF16 (no quantization)
- FP8_DYNAMIC (standard vLLM FP8)
- NVFP4/three-tier (NVFP4A16 on MLP + FP8 on attention + BF16 on norms)

SGLang is NOT supported for compressed-tensors mixed-precision checkpoints.

---

## Performance

Measured on NVIDIA RTX PRO 6000 Blackwell (96 GiB VRAM), model: Apertus v1.5 70B NVFP4 (48 GiB).
Results from [llmapibenchmark](https://github.com/onpremai/llmapibenchmark) with random prompts,
500-token decode, 200-token prefill workloads.

| Concurrency | Decode TPS | Prefill TPS | TTFT |
|-------------|-----------|-------------|------|
| 1 | 31 | 2500 | 0.3s |
| 4 | 104 | 2500 | 1.3s |
| 8 | 176 | 2500 | 2.7s |
| 20 | 294 | 2500 | 6.8s |

---

## Building

```bash
git clone https://github.com/onpremai/vllm-apertus-1p5.git
cd vllm-apertus-1p5
./build.sh [tag]
```

The build requires the CUDA toolkit and can take considerably longer when
compiler concurrency is limited for safety.

### Good to Know: CUDA Build Memory

vLLM defaults its native extension build to the number of available CPUs. On a
28-CPU, 62 GiB, no-swap host, this produced `ninja -j28` and launched many
parallel `nvcc`, `cicc`, `cudafe++`, and `cc1plus` processes. Individual CUDA
compiler processes used up to approximately 2.6 GiB of RAM. The host entered
severe memory-reclaim thrashing and required a hard power cycle.

## Target Hardware

This image compiles CUDA kernels for sm_80, sm_86, sm_89, sm_90a, sm_100a, and sm_120a.
This covers Ampere through Blackwell-ultra datacenter GPUs.

| Hardware | Arch | In this image? | Notes |
|----------|------|----------------|-------|
| H100 / H200 (Hopper) | SM90a | Yes | |
| B100 / B200 / GB200 (Grace Blackwell) | SM100a | Yes | Datacenter Blackwell with TMEM, WGMMA, DSMEM, NVSwitch. |
| B300 | SM100a | Yes | B300 is a B200 derivative, same architecture. |
| RTX PRO 6000 Blackwell (GB202) | SM100a | Yes | Primary target for this project. NVFP4 inference confirmed working. |
| NVIDIA GB300 (upcoming, code name Vera Rubin NEXT)| SM120a | Yes (forward-compatible) | sm_120a compile target included. If the real arch differs this table must be updated. |
| RTX 6000 Ada / RTX 4090 | SM89 | Yes | |
| A100 / A6000 (Ampere) | SM80 | Yes | |
| NVIDIA DGX Spark (GB10) | **SM121** | **No** | SM121 is not covered by sm_100a or sm_120a. Additionally, FP4 CUTLASS kernels produce silent garbage on SM121 (no TMEM, WGMMA, or DSMEM), making NVFP4-quantized checkpoints unusable. DGX Spark requires a separate build targeting SM121. See [DGX Spark SM121 reference](https://conselara.dev/notes/dgx-spark-gb10-hardware-reference/). |
| GB10 in Jetson / embedded form factors | SM121 | No | Same SM121 limitations as DGX Spark. |
| AMD GPUs (MI300X, etc.) | n/a | No | Requires vLLM ROCm build. |

Keep these build limits in the Dockerfile:

```dockerfile
ARG max_jobs=4
ARG nvcc_threads=8
ENV MAX_JOBS=${max_jobs}
ENV NVCC_THREADS=${nvcc_threads}
```

vLLM reads these variables in `setup.py`. `MAX_JOBS` controls Ninja
target-level parallelism (how many shared libraries compile concurrently).
With the CMake compile pool set to 1, each target serializes its individual
file compilations. `NVCC_THREADS` controls internal NVCC thread count.

On a 62 GiB no-swap host, `MAX_JOBS=4` uses approximately 15-24 GiB peak RAM
and completes in roughly 20-25 minutes. Do not remove or increase these limits
without measuring peak host RAM usage. An unbounded parallel build (`-j$(nproc)`)
can make a no-swap host unresponsive.

## Attribution

This project builds on the work of many contributors:

- **Swiss AI** (including [Oleg Lavrovsky (loleg)](https://github.com/loleg),
  [Cyrilvallez](https://github.com/Cyrilvallez)) -- Original
  [Apertus 1.5 model](https://huggingface.co/swiss-ai/Apertus-v1.5-70B),
  Apache 2.0
- **[blancsw](https://github.com/blancsw)** at Infomaniak -- GPU-native
  multimodal refactor, Emu3.5 vision encoder, WavTokenizer audio encoder,
  apertus tool parser
- **[Anunay-Yadav](https://github.com/Anunay-Yadav)** -- Upstream
  [PR #50496](https://github.com/vllm-project/vllm/pull/50496): GPU-native
  Apertus 1.5 multimodal encoders for upstream vLLM
- **[AryanAhadinia](https://github.com/AryanAhadinia)** -- Apertus reasoning parser
- **[robmsmt](https://github.com/robmsmt)** -- Double BOS-token fix
- **Neural Magic** -- [llm-compressor](https://github.com/vllm-project/llm-compressor)
  quantization tooling
- **onprem-ai** -- [llmapibenchmark](https://github.com/onpremai/llmapibenchmark)
  benchmarking tool

---
