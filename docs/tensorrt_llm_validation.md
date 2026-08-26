# TensorRT-LLM E007 Validation

## Purpose

`E007` verifies the saved-engine TensorRT-LLM workflow required by the original experiment matrix. A successful
result proves all three configured checks:

1. `tensorrt_llm_import`: the selected TensorRT-LLM environment imports on the allocated CUDA GPU.
2. `engine_build`: a pinned local TinyLlama checkpoint is converted into a saved engine with a provenance manifest.
3. `request_served`: the saved engine is loaded in a new runtime and serves deterministic measured requests.

E007 remains marked `deferred` until this workflow succeeds on the target H100. Local mocked tests validate the
control flow but are not GPU evidence.

## Supported Contract

The adapter implements `legacy_llm_engine_v1`, which uses the high-level TensorRT-LLM `LLM` API to build from a
local checkpoint, save an engine, reload that engine, and generate text. E007 pins TensorRT-LLM `0.19.0`; another
version fails at runtime because this saved-engine API is version-sensitive. A release that has removed saved-engine
support cannot satisfy the matrix's `engine_build` check.

The engine manifest records:

- immutable Hugging Face model name and revision;
- precision, tensor-parallel size, and build limits;
- TensorRT-LLM, PyTorch, CUDA, GPU, and compute-capability details;
- SHA-256 hashes for every saved engine file.

Prebuilt engines can use `engine_mode=validate_prebuilt`, but must provide a compatible manifest. E007 currently
uses `build_and_validate` so it proves the build environment as well as inference.

## Cluster Preparation

1. Create a TensorRT-LLM-specific environment or use a compatible NVIDIA container.
2. Place the exact TinyLlama revision from the E007 config in cluster-local storage. The resolved snapshot path must
   contain the full revision, or its root must contain `hpc_ai_perf_model_revision.txt` with that exact revision. Do
   not rely on an unpinned network download from a compute node.
3. Copy and edit the profile:

   ```bash
   cp slurm/profiles/tensorrt_llm_h100.example.json slurm/profiles/tensorrt_llm_h100.local.json
   ```

4. Replace the account, environment activation, model path, engine root, and CUDA module with site-valid values.
5. Confirm the selected TensorRT-LLM package exposes `tensorrt_llm.LLM`, `SamplingParams`, and `LLM.save()`.

The configured engine output includes `${SLURM_JOB_ID}`, preventing one job from overwriting another job's engine.
Existing non-empty build directories are rejected rather than deleted.

## Render and Run

Check the profile and render the dedicated E007 job:

```bash
SLURM_PROFILE=slurm/profiles/tensorrt_llm_h100.local.json make check-slurm-profile
SLURM_PROFILE=slurm/profiles/tensorrt_llm_h100.local.json make render-slurm-tensorrt-llm
sbatch slurm/generated/tensorrt_llm/E007.sbatch
```

E007 is rendered directly because deferred rows are intentionally excluded from bulk submission. Do not remove its
deferred marker before reviewing the first completed result.

## Acceptance Criteria

The result is acceptable when:

- the job runs on the requested H100 and records compute capability 9.0;
- the result status is `completed`;
- `tensorrt_llm_import`, `engine_build`, `engine_loaded`, and `request_served` are true;
- at least three measured requests and three generated tokens are recorded;
- the engine manifest exists and its file hashes validate;
- the result records the pinned model revision and TensorRT-LLM version;
- no CPU, PyTorch, or vLLM fallback was used.

After acceptance, remove `metadata.execution_status` and `metadata.defer_reason` from E007, regenerate the readiness
outputs, and commit the resulting activation separately. E131 and E179 remain deferred until the same engine contract
is integrated into the performance benchmark adapter. E155 remains deferred until a classification-compatible
contract is designed.
