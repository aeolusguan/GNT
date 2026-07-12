# Monotone Spectral NIS Design

## Objective

Improve final full-PGO MoGe reliability when different low-frequency DCT modes
have mixed quality. Replace one scene-wide scalar trust with ordered per-mode
trust for an ablation, while preserving the existing global K8 MoGe factor,
joint pose-scale objective, and runtime lifecycle.

The design is sequence-internal. It does not use GT, scene labels, validation
statistics, learned calibration, temporal subwindows, or localized MoGe
factors.

## Evidence

The 32-scene diagnostic contains 224 non-DC whitened components. MoGe is better
than graph-only for 114 components and worse for 110, confirming mixed mode
reliability. Raw independent per-mode NIS is too noisy: its trust direction
agrees with the GT oracle for only 52.7 percent of components.

Reliability has an ordered spectral structure. MoGe is useful more often at
low DCT frequencies than at high frequencies. This matches the intended system
roles: MoGe supplies global scale shape, while GeoNT relative measurements own
local scale consistency.

An offline linear proxy gives the following pre-ablation result:

| Policy | 32-scene mean whitened error energy | Wins vs scalar |
|---|---:|---:|
| Current scalar NIS | 0.04774 | baseline |
| Raw per-mode NIS | 0.04606 | 19/32 |
| Monotone spectral NIS | 0.04415 | 21/32 |

The proxy is evidence to run the ablation, not evidence to change the runtime
default. It was computed from the existing symmetric-whitened diagnostic;
ordered Cholesky whitening must therefore be evaluated by the actual replay
rather than assumed to reproduce the proxy values.

## Scope

- Apply only to final full-graph PGO with the existing K8 configuration.
- Keep marginalized local PGO at its current K2 scalar-NIS behavior.
- Keep the DCT basis, non-DC mode convention, NIS cutoff, LM schedule, pose
  Huber loss, edge weights, scale convention, and mode factor lifecycle
  unchanged.
- Freeze the monotone prior at the same first linearization where the current
  scalar prior is frozen.
- Do not add production config or CLI options during the ablation.
- Do not add temporal windows, adaptive mode count, scene trust, learned
  calibration, localized factors, or new graph states.

## Mathematical Design

Let the final graph mode covariance and innovation at the first linearization
be

\[
C = U H^{-1} U^T,
\qquad
\nu = U(x-m) + U\,dx_{\mathrm{graph}}.
\]

Here, the rows of \(U\) are the seven non-DC DCT modes in increasing frequency
order, \(k=1,\ldots,7\). The base Hessian \(H\) is exactly the Hessian already
used by current final NIS, including the fixed anchor constraint and excluding
the MoGe factor.

Regularize \(C\) with the existing covariance regularization, then compute the
ordered Cholesky factorization

\[
C = L L^T.
\]

Compute conditional whitened innovations

\[
z = L^{-1}\nu,
\qquad
r_k = z_k^2.
\]

Unlike symmetric ZCA whitening, Cholesky whitening preserves the DCT frequency
order. Component \(z_k\) measures mode \(k\) after conditioning on all lower
frequency modes.

Project the raw scores onto the nondecreasing cone with equal-weight isotonic
regression:

\[
\hat r =
\arg\min_{s_1\le s_2\le\cdots\le s_7}
\sum_{k=1}^{7}(s_k-r_k)^2.
\]

Use the pool-adjacent-violators algorithm (PAVA). It is deterministic and adds
no tunable parameter.

Convert the projected scores to ordered trust values using the existing NIS
cutoff \(c\):

\[
\alpha_k =
\operatorname{clip}\left(
\frac{c}{c+\hat r_k},
10^{-6},
1-10^{-4}
\right).
\]

This guarantees

\[
\alpha_1\ge\alpha_2\ge\cdots\ge\alpha_7.
\]

Construct the MoGe mode prior covariance in the same ordered conditional
coordinates:

\[
R_m =
L\,
\operatorname{diag}\left(
\frac{1-\alpha_k}{\alpha_k}
\right)
L^T.
\]

When all \(\alpha_k\) are equal, this exactly reduces to the current scalar
form

\[
R_m = \frac{1-\alpha}{\alpha}C.
\]

The existing mode precision, Woodbury correction, mode quadratic cost, and LM
candidate evaluation consume \(R_m\) without further changes.

## Ablation Implementation Boundary

The ablation may introduce one private final-optimizer experiment switch and a
dedicated artifact replay script. It must not add a production config field.
Torch and CUDA-Eigen must implement the same mathematics. The local solver must
continue calling the current scalar helper.

After evaluation:

- If the method fails the acceptance criteria, delete the experiment switch,
  monotone helper, tests specific to the discarded path, and replay script if
  it has no remaining diagnostic value.
- If the method passes, replace the final scalar helper with the monotone
  implementation and remove the temporary policy branch. Do not retain two
  permanent final-NIS policies.

## Reporting

For every scene, record:

- node and edge counts;
- raw conditional NIS values \(r_k\);
- isotonic scores \(\hat r_k\);
- ordered trust values \(\alpha_k\);
- number of PAVA blocks;
- current scalar NIS score and alpha;
- ATE RMSE;
- GT-aligned log-scale shape RMSE;
- all, temporal, and nonlocal edge relative-scale RMSE;
- accepted LM iterations;
- final PGO runtime.

The replay output must contain JSON and CSV summaries and must identify the
policy used by every row.

## Validation

### Mathematical tests

- Cholesky whitening produces identity covariance.
- PAVA output is nondecreasing.
- Already nondecreasing scores are unchanged.
- Ordered alpha values are nonincreasing.
- Equal alpha values reproduce the current scalar prior covariance.
- The prior covariance is symmetric positive definite after the existing
  regularization and clamps.

### Backend tests

- Torch and CUDA-Eigen produce matching Cholesky scores, isotonic scores,
  alpha values, and optimized state on a synthetic graph.
- Scalar mode remains numerically unchanged when the experiment switch is
  disabled.
- Marginalized local PGO output and lifecycle remain unchanged.

### Replay order

1. P011 and P007 targeted replay.
2. Complete 32-scene artifact replay.
3. Inspect P008 and neighborhood/Hard/P017 explicitly because the proxy
   identifies them as regression risks.

## Acceptance Criteria

All criteria must pass:

### Target scenes

- P007 ATE RMSE is lower than the scalar baseline 2.07620.
- P007 scale-shape RMSE is lower than the scalar baseline 0.10827.
- P011 ATE regresses by no more than 1 percent from the scalar baseline.
- P011 scale-shape RMSE regresses by no more than 0.001 absolute.

### 32-scene aggregate

- Mean ATE does not regress.
- Median ATE does not regress.
- Mean scale-shape RMSE does not regress.
- At least 18 of 32 scenes improve in ATE.
- No scene has more than 3 percent relative ATE regression.

### Runtime

- Mean final-PGO runtime increases by less than 1 percent relative to scalar
  NIS.

Runtime comparison uses one untimed warmup followed by three timed repetitions
per policy and scene. The per-scene median enters the aggregate, avoiding a
decision based on one noisy short solve.

Failure of any criterion keeps scalar NIS as the final runtime method and
triggers experiment-path cleanup.
