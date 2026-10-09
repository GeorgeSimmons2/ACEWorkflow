# Reproduction report

Generated 2026-10-02 13:35 by `constrained_optimization/verify/compare_all.jl`.

Tiers, and why they differ — see `lib_verify.jl` for the measurements behind them:

- `exact` — no QP in the member path (a seeded draw, or a closed-form correction).
  Must match to machine precision; anything else is a bug, not solver noise.
- `converged` — one well-posed QP. The iterate path may differ, the optimum does not.
- `statistical` — a cutting-plane cascade. Members and their order are not
  reproducible; the distribution is. Never quote a member index.


| quantity | tier | verdict | detail |
|---|---|---|---|
| W positive-core θ | `converged` | **FAIL** | max|Δ| = 1.688e+00  median|Δ| = 3.734e-05  ‖Δ‖/‖θ‖ = 1.434e-01  (tol 1.0e-02) |
| Al_12 θ_mean (ncell4 densek) | `converged` | **PASS** | max|Δ| = 5.454e-06  median|Δ| = 4.740e-08  ‖Δ‖/‖θ‖ = 5.021e-06  (tol 1.0e-02) |
| Al_12 committee_repaired (30) | `statistical` | **PASS** | n=30  worst z(mean) = 0.88, worst z(std) = 1.10  (threshold 4.5 σ of sampling noise) |
| Al_12 committee_rejection (30) | `statistical` | **FAIL** | n=30  worst z(mean) = 3.43, worst z(std) = 4.56  (threshold 4.5 σ of sampling noise) |
| Al_12 cutting-plane cloud, min ω | `statistical` | **PASS** | n=73479  min +0.1495→+0.1482  median +0.1669→+0.1666  max +0.4088→+0.4099 THz  (rel 0.0034, tol 0.010) |
| Al_12 rejection committee (full cloud) | `statistical` | **PASS** | n=30  worst z(mean) = 2.09, worst z(std) = 2.98  (threshold 4.5 σ of sampling noise) |
| Al_12 naive ensemble (30 × 91) | `converged` | **FAIL** | max|Δ| = 9.638e-02  median|Δ| = 1.443e-03  ‖Δ‖/‖θ‖ = 5.681e-03  (tol 1.0e-02) |
| Al_12 naive relaxed a | `converged` | **PASS** | n=30  min +3.9793→+3.9538  median +4.0873→+4.0795  max +4.2970→+4.3005 Å  (rel 0.0059, tol 0.010) |
| Al_16 unconstrained, native min ω (grid-matched) | `converged` | **FAIL** | n=20  min -5.6776→-5.2516  median +0.4014→+0.3937  max +0.4471→+0.4436 THz  (rel 0.0750, tol 0.010) |
| Al_16 constrained, native min ω (grid-matched) | `converged` | **FAIL** | n=20  min +0.3748→+0.3360  median +0.4373→+0.4291  max +0.4937→+0.4943 THz  (rel 0.0786, tol 0.010) |
| NPT multi-volume committee θ_mean | `converged` | **FAIL** | max|Δ| = 3.495e-02  median|Δ| = 4.452e-05  ‖Δ‖/‖θ‖ = 1.324e-02  (tol 1.0e-02) |
| NPT multi-volume committee_rejection (30) | `statistical` | **PASS** | n=30  worst z(mean) = 3.57, worst z(std) = 2.92  (threshold 4.5 σ of sampling noise) |
| NPT a_eq committee θ_mean | `converged` | **FAIL** | max|Δ| = 1.048e-01  median|Δ| = 6.973e-05  ‖Δ‖/‖θ‖ = 5.355e-02  (tol 1.0e-02) |
| NPT a_eq committee_rejection (30) | `statistical` | **PASS** | n=30  worst z(mean) = 4.01, worst z(std) = 2.89  (threshold 4.5 σ of sampling noise) |

7 passed, 7 failed, 0 not yet produced.
