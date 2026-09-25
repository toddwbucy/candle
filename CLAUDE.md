# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Candle is a minimalist ML framework for Rust (CPU, CUDA, Metal, WASM). This clone is the
`toddwbucy/candle` fork of `huggingface/candle`; see "Fork workflow" below.

## Commands

Mirror CI (`.github/workflows/rust-ci.yml`) before pushing:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --tests --examples --benches -- -D warnings
cargo test --workspace
```

- Single crate / single test: `cargo test -p candle-core --test tensor_tests matmul` (test files live in `<crate>/tests/*.rs`).
- GPU backends are opt-in features: `--features cuda`, `cudnn`, `metal`, `mkl`, `accelerate`. Without them, CUDA/Metal test variants are compiled out.
- Examples (in `candle-examples/examples/<name>/`): `cargo run --example quantized --release [--features cuda|cudnn|flash-attn|metal]`.
- Excluded crates (`candle-flash-attn`, `candle-flash-attn-v3`, `candle-kernels`, `candle-metal-kernels`, `candle-onnx`, `candle-book`) are not workspace members; build/test them from their own directory (e.g. `cd candle-onnx && cargo test`). They are still pulled in as path deps via features.
- `make clean-ptx` forces CUDA kernels to rebuild (stale PTX after editing `.cu` files).
- Python bindings: `cd candle-pyo3 && maturin develop -r && python test.py`; regenerate type stubs with `python stub.py` (`--check` to verify).
- `.cargo/config.toml` sets `-C target-cpu=native` for all builds; binaries are not portable across CPUs.
- A `release-with-debug` profile exists for profiling.

## Architecture

**Tensor → Storage → Backend layering (candle-core).**
- `Tensor` (`tensor.rs`) is an `Arc` around storage + `Layout` (shape/strides/offset, `layout.rs`). Views like `narrow`/`transpose` only change the layout; ops must handle non-contiguous layouts or call `.contiguous()`.
- `Storage` (`storage.rs`) is an enum `Cpu | Cuda | Metal` that dispatches each op to the backend; `Device` (`device.rs`) mirrors it.
- Every backend implements the `BackendStorage` / `BackendDevice` traits (`backend.rs`): `cpu_backend/`, `cuda_backend/`, `metal_backend/`. When a feature is disabled, `dummy_cuda_backend.rs` / `dummy_metal_backend.rs` stand in and return errors, so code compiles on every platform.
- Elementwise ops are defined once as `UnaryOpT`/`BinaryOpT` impls in `op.rs` and reused by all backends.
- Adding or changing an op typically touches: `tensor.rs` (API + recording the `Op` for autograd), `storage.rs` (dispatch), each backend, `backprop.rs` (gradient), plus kernels in `candle-kernels/src/*.cu` (CUDA, compiled to PTX by `build.rs` via `cudaforge`; MoE/GGUF matmul kernels are compiled statically instead) and `candle-metal-kernels` (Metal). `custom_op.rs` provides `CustomOp1/2/3` for ops outside core (used e.g. by flash-attn and many `candle-nn::ops`).
- CPU SIMD paths live in `cpu/` (avx, neon, simd128); quantization (GGML/GGUF k-quants, per-backend matmul) lives in `quantized/`.
- Tests use `test_device!(fn, cpu_name, cuda_name, metal_name)` from `test_utils.rs` to run one test body on every enabled device.

**candle-nn** — layers (`Linear`, `Embedding`, norms, conv, RNN), `ops` (softmax, rms_norm, sdpa…), `rotary_emb`, `kv_cache`, optimizers. Weights are loaded through `VarBuilder` (`var_builder.rs`), which abstracts safetensors/npz/pth/`VarMap` sources and handles dtype/device placement and name prefixes (`vb.pp("layer")`). Layers implement `candle::Module` (`forward`) or `ModuleT` (`forward_t` with a train flag).

**candle-transformers** — model implementations in `src/models/` (one file or dir per architecture; `quantized_*` variants use `quantized_var_builder` / `quantized_nn` over GGUF). `generation/` has `LogitsProcessor` sampling. Models are driven by the matching example in `candle-examples`, which handles hub download (`hf-hub`), tokenization and the generation loop.

## Fork workflow

This is the `toddwbucy/candle` fork. The downstream consumer is WeaverTools
(`toddwbucy/WeaverTools`), which pins candle crates by git rev. Two goals drive the
branch model: stay as close to upstream as possible, and never let fork-only material
leak into an upstream PR before it is meant to be submitted.

**Branches**

- `main` is a pure mirror of `huggingface/candle`. It moves only by fast-forward to
  the `upstream/main` tip. Nothing is ever committed to fork `main`: no features, no
  config, no docs. Synced 2026-09-25 to `66a8cf18`.
- `integration` is the fork's working branch and the GitHub default branch. Every
  `feat/*` branch is merged here via PR. WeaverTools is meant to pin commits on
  `integration`. Fork-only files (`CLAUDE.md`, `.coderabbit.yaml`, handoff notes) live
  here and only here.
- `feat/*` branches are cut from `main`, never from `integration`. That is the leak
  guard: a branch cut from `main` can only ever diff against upstream by its own
  commits. Each `feat/*` is merged into `integration` for WeaverTools and, when ready,
  opened as a PR against `huggingface/candle:main` directly from the `feat/*` branch.
  Fork `main` is not involved in upstream submission.
- Fork-only files (`CLAUDE.md`, `.coderabbit.yaml`, handoff notes) are committed
  directly to `integration`; no PR is needed for docs and config. Fork-only code changes
  (reverts of withdrawn features, sync merges) go on a branch cut from `integration` and
  PR'd back to it. Neither ever touches `main` lineage, so neither can appear in an
  upstream diff.
- `weaver/*` branches are frozen pin-holders for WeaverTools revs that predate this
  model. Do not move or delete them; retiring one is decided on the WeaverTools side
  when it repins to `integration`.

**Upstream PRs**

- Before opening or updating an upstream PR, confirm the diff carries no agent or
  review tooling files. This must print nothing:
  `git diff --name-only upstream/main...HEAD | grep -Ei 'claude\.md|agents\.md|coderabbit|\.cursor|copilot'`
- Posting to `huggingface/candle` (comments, PR descriptions, force-pushes of a PR
  branch) is the operator's decision per action. Draft the text, show it, wait.

**Fork PRs**

- Every change to `integration` goes through a PR on `toddwbucy/candle` with a
  CodeRabbit review and an independent sub-agent review. Nothing is pushed straight to
  `main`, `integration` or a `weaver/*` branch.
- Reverting a feature (e.g. after withdrawing its upstream PR) is a `git revert -m 1`
  of its merge commit on `integration`, via PR. No history rewriting.

**Remotes and sync**

- `upstream` = `https://github.com/huggingface/candle`. Add it if missing (`git remote
  add upstream ...`) and fetch before measuring anything against upstream.
- Syncing `main` is `git fetch upstream && git push origin <upstream-ref>:main`
  (fast-forward only). Then merge `main` into `integration` via PR.

**Records**

- Use ASCII and absolute dates in anything written for the record (PR text, handoff
  notes, issue comments). Cross-repo status is reported on the WeaverTools tracking
  issue, not by opening new issues.
