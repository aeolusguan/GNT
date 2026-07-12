# Monotone Spectral NIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a final-only Monotone Spectral NIS ablation that replaces one scalar MoGe trust with frequency-ordered K8 trust, then remove the losing policy path.

**Architecture:** Add a small shared mathematical helper for ordered Cholesky whitening, PAVA, and prior covariance construction. Thread one private ablation switch through only the final Torch and CUDA-Eigen joint PGO backends, leaving marginalized local K2 untouched. A dedicated artifact replay script compares graph-only, scalar NIS, and monotone NIS on P011/P007 and all 32 scenes.

**Tech Stack:** Python, PyTorch, LieTorch, existing CUDA-Eigen PGO extension, NumPy, TartanAir artifact replay.

## Global Constraints

- Implement the confirmed design in `docs/superpowers/specs/2026-07-12-monotone-spectral-nis-design.md` exactly.
- Apply monotone NIS only to final full-graph K8 PGO.
- Do not modify marginalized local K2 NIS.
- Do not change DCT modes, NIS cutoff, LM behavior, covariance regularization, edge weights, scale convention, or mode-factor freezing lifecycle.
- Do not add production config or CLI fields.
- Do not modify CUDA/C++ sources; the existing multi-RHS solver is sufficient.
- Preserve unrelated dirty worktree changes.
- After every local Python edit, narrow-sync the changed `.py` files to A6000 and verify SHA-256 parity before remote tests.
- If the ablation fails any acceptance criterion, remove the entire monotone optimizer path. If it passes, replace final scalar NIS and remove the temporary policy branch.

---

### Task 1: Ordered NIS Mathematics

**Files:**
- Modify: `src/geont_runtime/slam/pgo/common.py:210-235`
- Create: `tests/test_pgo_monotone_spectral_nis.py`

**Interfaces:**
- Consumes: regularized SPD mode covariance `covariance: torch.Tensor`, mode innovation `innovation: torch.Tensor`, and positive `nis_cutoff: float`.
- Produces: `_isotonic_non_decreasing(values) -> tuple[torch.Tensor, int]` and `_moge_monotone_mode_prior_covariance(covariance, innovation, nis_cutoff) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]`.
- The monotone helper returns `(prior_covariance, raw_conditional_nis, isotonic_nis, alpha, pava_block_count)`.

- [ ] **Step 1: Write failing PAVA and covariance tests**

Add tests that express the public mathematical contract without depending on an optimizer:

```python
import torch

from geont_runtime.slam.pgo.common import (
    _isotonic_non_decreasing,
    _moge_monotone_mode_prior_covariance,
)


def test_isotonic_projection_pools_adjacent_violations():
    values = torch.tensor([0.02, 0.12, 0.48, 0.01, 0.01, 0.06, 0.06])
    projected, block_count = _isotonic_non_decreasing(values)
    assert torch.all(projected[1:] >= projected[:-1])
    assert block_count < values.numel()
    assert torch.allclose(
        projected,
        torch.tensor([0.02, 0.12, 0.124, 0.124, 0.124, 0.124, 0.124]),
        atol=1e-6,
    )


def test_isotonic_projection_keeps_ordered_values():
    values = torch.tensor([0.01, 0.02, 0.04, 0.08])
    projected, block_count = _isotonic_non_decreasing(values)
    assert torch.equal(projected, values)
    assert block_count == values.numel()


def test_monotone_prior_has_ordered_alpha_and_is_spd():
    matrix = torch.tensor(
        [[2.0, 0.4, 0.1], [0.4, 1.5, 0.2], [0.1, 0.2, 1.0]],
        dtype=torch.float64,
    )
    innovation = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    prior, raw, projected, alpha, blocks = _moge_monotone_mode_prior_covariance(
        matrix, innovation, 0.01
    )
    assert torch.all(projected[1:] >= projected[:-1])
    assert torch.all(alpha[1:] <= alpha[:-1])
    assert torch.all(torch.linalg.eigvalsh(prior) > 0)
    assert torch.allclose(prior, prior.T, atol=1e-12)
    assert blocks >= 1
```

Add a scalar-reduction test by constructing `R = L @ diag((1-alpha)/alpha) @ L.T` with identical alpha values and comparing against `((1-alpha)/alpha) * covariance`.

- [ ] **Step 2: Run the tests and verify RED**

Local environment lacks Torch/PyTest, so first run syntax checking locally, then sync the test and run the import on A6000:

```bash
python -m py_compile tests/test_pgo_monotone_spectral_nis.py
rsync -azR tests/test_pgo_monotone_spectral_nis.py A6000:/data/disk_7t/tongfan/GNT/
bash scripts/a6000-run python -c 'import runpy; runpy.run_path("tests/test_pgo_monotone_spectral_nis.py")'
```

Expected: import fails because the two helpers do not exist.

- [ ] **Step 3: Implement deterministic equal-weight PAVA**

Add a direct seven-value implementation in `common.py`. Do not add a dependency:

```python
def _isotonic_non_decreasing(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    samples = values.to(device="cpu", dtype=torch.float64).tolist()
    blocks: list[list[float | int]] = []
    for index, value in enumerate(samples):
        blocks.append([index, index, float(value), 1])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if float(left[2]) / int(left[3]) <= float(right[2]) / int(right[3]):
                break
            blocks[-2:] = [[
                int(left[0]),
                int(right[1]),
                float(left[2]) + float(right[2]),
                int(left[3]) + int(right[3]),
            ]]
    projected = [0.0] * len(samples)
    for start, end, total, count in blocks:
        mean = float(total) / int(count)
        projected[int(start) : int(end) + 1] = [mean] * (int(end) - int(start) + 1)
    return values.new_tensor(projected), len(blocks)
```

The one tiny GPU-to-CPU transfer is acceptable because this runs once per final PGO invocation, and the production backend already performs CPU Eigen solves.

- [ ] **Step 4: Implement ordered Cholesky NIS and prior covariance**

Add:

```python
def _moge_monotone_mode_prior_covariance(
    covariance: torch.Tensor,
    innovation: torch.Tensor,
    nis_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    chol = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(
        chol,
        innovation[:, None],
        upper=False,
    ).squeeze(-1)
    raw_nis = whitened.square()
    isotonic_nis, block_count = _isotonic_non_decreasing(raw_nis)
    alpha = float(nis_cutoff) / (float(nis_cutoff) + isotonic_nis)
    alpha = alpha.clamp(min=1e-6, max=1.0 - 1e-4)
    ratio = (1.0 - alpha) / alpha
    prior = (chol * ratio.sqrt()[None, :]) @ (chol * ratio.sqrt()[None, :]).T
    prior = 0.5 * (prior + prior.T)
    return prior, raw_nis, isotonic_nis, alpha, block_count
```

Do not add another jitter. The caller already passes `_regularize_mode_covariance(...)` output.

- [ ] **Step 5: Sync immediately and verify GREEN on A6000**

```bash
rsync -azR src/geont_runtime/slam/pgo/common.py tests/test_pgo_monotone_spectral_nis.py A6000:/data/disk_7t/tongfan/GNT/
shasum -a 256 src/geont_runtime/slam/pgo/common.py tests/test_pgo_monotone_spectral_nis.py
bash scripts/a6000-run sha256sum src/geont_runtime/slam/pgo/common.py tests/test_pgo_monotone_spectral_nis.py
bash scripts/a6000-run python -c 'import runpy; ns=runpy.run_path("tests/test_pgo_monotone_spectral_nis.py"); [ns[name]() for name in ns if name.startswith("test_")]'
```

Expected: all pure mathematical tests pass and hashes match.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/geont_runtime/slam/pgo/common.py tests/test_pgo_monotone_spectral_nis.py
git commit -m "test: define monotone spectral NIS mathematics"
```

---

### Task 2: Final Torch and CUDA-Eigen Ablation Path

**Files:**
- Modify: `src/geont_runtime/slam/pgo/optimizer.py:99-270`
- Modify: `src/geont_runtime/slam/pgo/torch_backend.py:690-905`
- Modify: `src/geont_runtime/slam/pgo/cuda_eigen.py:410-610`
- Modify: `tests/test_pgo_monotone_spectral_nis.py`

**Interfaces:**
- Consumes: private `moge_mode_monotone_nis: bool = False` from the public optimizer, valid only when `mode="se3_scale"` and `moge_mode_nis=True`.
- Produces: unchanged `Sim3PGOResult` plus monotone diagnostic fields in `info`.
- Diagnostic fields: `moge_mode_raw_nis`, `moge_mode_isotonic_nis`, `moge_mode_alpha_values`, and `moge_mode_pava_blocks`.
- Existing scalar fields remain floats. For monotone runs, set `moge_mode_nis_score` to the median raw conditional NIS and `moge_mode_alpha` to the median ordered alpha for concise summaries.

- [ ] **Step 1: Add failing Torch behavior test**

Reuse the synthetic K8 graph pattern in `tests/test_pgo_scale_gauge.py`. Run scalar and monotone final Torch solves from identical tensors. Assert:

```python
assert monotone.info["moge_mode_alpha_values"] == sorted(
    monotone.info["moge_mode_alpha_values"], reverse=True
)
assert len(monotone.info["moge_mode_raw_nis"]) == 7
assert len(monotone.info["moge_mode_isotonic_nis"]) == 7
assert "moge_mode_alpha_values" not in scalar.info
```

Also run scalar twice, once without the new keyword and once with
`moge_mode_monotone_nis=False`, and require exactly matching state/info.

- [ ] **Step 2: Verify RED on A6000**

Sync the test immediately and call its new test function with `runpy`.

Expected: `optimize_sim3_pose_graph()` rejects the unknown keyword.

- [ ] **Step 3: Thread the private switch through final PGO only**

In `optimizer.py`:

```python
moge_mode_monotone_nis: bool = False,
```

Validate that monotone requires `moge_mode_nis=True` and `mode="se3_scale"`.
Pass it only to `_optimize_se3_scale_pose_graph` or
`_optimize_se3_scale_pose_graph_cuda`. Do not pass it to staged, rotation-only,
or marginalized local code.

- [ ] **Step 4: Select and freeze the prior in the Torch backend**

Import `_moge_monotone_mode_prior_covariance`. At the existing
`if mode_prior_covariance is None:` block:

```python
if moge_mode_monotone_nis:
    (
        mode_prior_covariance,
        mode_raw_nis,
        mode_isotonic_nis,
        mode_alpha_values,
        mode_pava_blocks,
    ) = _moge_monotone_mode_prior_covariance(
        mode_covariance,
        innovation,
        moge_mode_nis_cutoff,
    )
    mode_nis_score = float(mode_raw_nis.median().cpu())
    mode_alpha = float(mode_alpha_values.median().cpu())
else:
    mode_prior_covariance, mode_nis_score, mode_alpha = _moge_mode_prior_covariance(
        mode_covariance,
        innovation,
        moge_mode_nis_cutoff,
    )
```

Freeze all arrays with the prior. Add their Python-list forms to `info` only
for monotone runs. Keep the existing precision, correction, and cost code.

- [ ] **Step 5: Mirror the same selection in CUDA-Eigen**

Apply the same helper call and info fields in `cuda_eigen.py`. Do not change the
Eigen solver, multi-RHS inputs, weighted CUDA blocks, or LM retries.

- [ ] **Step 6: Add CUDA-Eigen versus Torch regression**

Run identical synthetic monotone K8 solves on Torch and CUDA-Eigen. Require:

```python
assert torch.allclose(cuda.log_scales, reference.log_scales, atol=2e-4, rtol=2e-4)
assert torch.allclose(cuda.poses, reference.poses, atol=2e-4, rtol=2e-4)
assert torch.allclose(
    torch.tensor(cuda.info["moge_mode_alpha_values"]),
    torch.tensor(reference.info["moge_mode_alpha_values"]),
    atol=2e-4,
    rtol=2e-4,
)
```

- [ ] **Step 7: Sync all changed Python files and run targeted A6000 tests**

```bash
rsync -azR \
  src/geont_runtime/slam/pgo/common.py \
  src/geont_runtime/slam/pgo/optimizer.py \
  src/geont_runtime/slam/pgo/torch_backend.py \
  src/geont_runtime/slam/pgo/cuda_eigen.py \
  tests/test_pgo_monotone_spectral_nis.py \
  A6000:/data/disk_7t/tongfan/GNT/
```

Verify hashes, `py_compile`, the new direct test functions, all zero-argument
tests in `tests/test_pgo_scale_gauge.py`, and all zero-argument tests in
`tests/test_marginalized_local_pgo.py`.

Expected: scalar state is unchanged, monotone alpha is ordered, Torch/CUDA
match, and local lifecycle tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  src/geont_runtime/slam/pgo/common.py \
  src/geont_runtime/slam/pgo/optimizer.py \
  src/geont_runtime/slam/pgo/torch_backend.py \
  src/geont_runtime/slam/pgo/cuda_eigen.py \
  tests/test_pgo_monotone_spectral_nis.py
git commit -m "feat: add final monotone spectral NIS ablation"
```

---

### Task 3: Artifact Replay and Reporting

**Files:**
- Create: `scripts/ablate_pgo_monotone_spectral_nis.py`
- Create: `tests/test_ablate_pgo_monotone_spectral_nis.py`

**Interfaces:**
- Consumes: artifacts under `outputs/tartanair_marginalized_local_pgo_32`, TartanAir GT, and the private optimizer switch from Task 2.
- Produces: `summary.json`, `scenes.csv`, and `modes.csv` under `outputs/pgo_monotone_spectral_nis_ablation_32scene`.
- Policies are exactly `graph_only`, `scalar_nis`, and `monotone_spectral_nis`.

- [ ] **Step 1: Write failing summary/report tests**

Test pure helpers with synthetic rows:

- policy names are fixed and ordered;
- paired ATE and scale deltas use scalar NIS as baseline;
- monotone mode rows contain seven raw scores, seven isotonic scores, seven
  ordered alphas, and a positive PAVA block count;
- acceptance evaluation checks every criterion from the spec;
- one scene with 3.1 percent ATE regression rejects the candidate even if means
  improve.

- [ ] **Step 2: Verify RED on A6000**

Sync the test and confirm import failure because the ablation script does not
exist.

- [ ] **Step 3: Implement the replay script by following existing artifact scripts**

Reuse artifact/GT helpers from
`scripts/ablate_pgo_nonlocal_scale_confidence.py` and metric helpers from
`scripts/evaluate_tartanair_pgo.py`. Do not duplicate optimizer mathematics.

For each scene:

1. Load pose/depth artifacts and GT once.
2. Warm up each policy once without timing.
3. Run each policy three timed times from identical initial tensors.
4. Use the median runtime and the deterministic result from the final repeat.
5. Compute trajectory, scale-shape, and temporal/nonlocal/all edge-scale RMSE.
6. Write one scene row per policy and seven mode rows for the monotone policy.

Expose only ablation-script arguments:

```text
--artifact-root
--tartanair-root
--split
--output
--scene (repeatable)
--max-scenes
--device
```

Keep fixed research protocol values in code: final K8, cutoff 0.01, damping
1e-3, pose Huber 0.05, 12 iterations, and 5 LM attempts.

- [ ] **Step 4: Implement explicit acceptance output**

The final JSON must contain:

```json
{
  "acceptance": {
    "passed": false,
    "checks": {
      "p007_ate_improved": false,
      "p007_scale_improved": false,
      "p011_ate_within_1pct": true,
      "p011_scale_within_0p001": true,
      "mean_ate_not_worse": false,
      "median_ate_not_worse": true,
      "mean_scale_not_worse": true,
      "ate_wins_at_least_18": true,
      "max_ate_regression_within_3pct": false,
      "runtime_overhead_below_1pct": true
    }
  }
}
```

Compute checks from measured rows; do not hard-code outcomes.

- [ ] **Step 5: Sync and run a one-scene smoke test**

Run P011 with `--scene abandonedfactory/abandonedfactory/Easy/P011`. Verify
three policies, seven mode rows, ordered alpha, scalar baseline reproduction,
and complete files.

- [ ] **Step 6: Run P007 targeted replay**

Run P007 separately and inspect ATE, scale RMSE, raw scores, PAVA pooling, and
ordered alpha before spending time on the full replay.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/ablate_pgo_monotone_spectral_nis.py tests/test_ablate_pgo_monotone_spectral_nis.py
git commit -m "test: add monotone spectral NIS replay"
```

---

### Task 4: 32-Scene Decision and Mandatory Cleanup

**Files:**
- Read: `outputs/pgo_monotone_spectral_nis_ablation_32scene/summary.json`
- Conditionally modify/delete the files touched in Tasks 1-3 according to the measured decision.

**Interfaces:**
- Consumes: completed 32-scene JSON/CSV outputs.
- Produces: one clean final code path: either restored scalar final NIS or monotone final NIS with no experiment switch.

- [ ] **Step 1: Run the complete A6000 replay**

```bash
bash scripts/a6000-run bash -lc \
  'CUDA_VISIBLE_DEVICES=2 python scripts/ablate_pgo_monotone_spectral_nis.py \
  --output outputs/pgo_monotone_spectral_nis_ablation_32scene'
```

Do not change thresholds after seeing results.

- [ ] **Step 2: Copy outputs locally and inspect all acceptance checks**

```bash
rsync -az \
  A6000:/data/disk_7t/tongfan/GNT/outputs/pgo_monotone_spectral_nis_ablation_32scene/ \
  outputs/pgo_monotone_spectral_nis_ablation_32scene/
```

Report P011, P007, P008, neighborhood/Hard/P017, aggregate means/medians,
wins/losses, maximum regression, runtime, and the generated acceptance object.

- [ ] **Step 3A: If any criterion fails, remove the experiment path**

Use `apply_patch` to remove:

- `_isotonic_non_decreasing` and `_moge_monotone_mode_prior_covariance` if they
  have no remaining diagnostic use;
- `moge_mode_monotone_nis` from the optimizer and both final backends;
- monotone-only tests;
- the replay script if it has no continuing value.

Retain output artifacts and the design/plan documents. Narrow-sync restored
Python files and rerun scalar final/local tests.

- [ ] **Step 3B: If every criterion passes, make monotone final NIS the single path**

Use `apply_patch` to:

- call `_moge_monotone_mode_prior_covariance` unconditionally in final Torch
  and CUDA-Eigen NIS;
- remove the private experiment switch and scalar/monotone branch from final
  optimizer code;
- keep `_moge_mode_prior_covariance` only for marginalized local K2 scalar NIS;
- retain useful ordered-alpha diagnostics without adding config.

Narrow-sync and rerun all targeted tests.

- [ ] **Step 4: Final verification**

Run locally:

```bash
python -m compileall -q src/geont_runtime/slam/pgo
git diff --check
```

If the replay script and its test remain after the cleanup decision, also run
`python -m py_compile scripts/ablate_pgo_monotone_spectral_nis.py tests/test_ablate_pgo_monotone_spectral_nis.py`.

Run on A6000 after hash verification:

- monotone/scalar mathematical tests appropriate to the retained path;
- Torch/CUDA equality;
- `tests/test_pgo_scale_gauge.py` zero-argument tests;
- `tests/test_marginalized_local_pgo.py` zero-argument tests;
- one retained-method P011 smoke replay.

- [ ] **Step 5: Commit the cleanup decision**

Stage only files from this work and commit either:

```bash
git commit -m "revert: remove monotone spectral NIS ablation"
```

or:

```bash
git commit -m "feat: use monotone spectral NIS in final PGO"
```

The final report must state the measured decision, code path retained, A6000
test evidence, performance impact, and output paths.
