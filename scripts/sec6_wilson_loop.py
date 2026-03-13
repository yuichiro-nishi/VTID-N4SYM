"""
Section 6.2: 1/2-BPS Circular Wilson Loop in N=4 SYM
=====================================================
Computes:
  (A) Finite-N exact formula (U(N) Gaussian matrix model, Pestun localization)
      <W>_{U(N)} = (1/N) L^1_{N-1}(-lambda/(4N)) exp(lambda/(8N))
  (B) Planar large-N limit
      <W>_{planar} = (2/sqrt(lambda)) I_1(sqrt(lambda))
  (C) Weak/strong coupling checks
  (D) Mutual information estimate (CFT vacuum, two disjoint intervals)

References:
  [28] Maldacena, Phys. Rev. Lett. 80 (1998) 4859
  [29] Erickson-Semenoff-Zarembo, Nucl. Phys. B 582 (2000) 155
  [30] Drukker-Gross, J. Math. Phys. 42 (2001) 2896
  [31] Pestun, Commun. Math. Phys. 313 (2012) 71
"""

import json
import sys
import numpy as np
from scipy.special import iv, eval_genlaguerre
from datetime import datetime

# ============================================================
# Core: Wilson loop formulas
# ============================================================

def wilson_finite_N(lam, N):
    """Finite-N exact result for 1/2-BPS circular Wilson loop (U(N)).
    <W>_{U(N)} = (1/N) L^1_{N-1}(-lambda/(4N)) exp(lambda/(8N))
    where L^alpha_n is the generalized Laguerre polynomial.
    """
    x = -lam / (4.0 * N)
    lag = eval_genlaguerre(N - 1, 1, x)
    return (1.0 / N) * lag * np.exp(lam / (8.0 * N))


def wilson_planar(lam):
    """Planar large-N limit: <W> = (2/sqrt(lambda)) I_1(sqrt(lambda))"""
    sl = np.sqrt(lam)
    return (2.0 / sl) * iv(1, sl)


def wilson_weak(lam):
    """Weak coupling: <W> = 1 + lambda/8 + lambda^2/192 + ..."""
    return 1.0 + lam / 8.0 + lam**2 / 192.0


def wilson_strong(lam):
    """Strong coupling asymptotic of planar formula:
    <W> ~ sqrt(2/pi) lambda^{-3/4} exp(sqrt(lambda))
    """
    sl = np.sqrt(lam)
    return np.sqrt(2.0 / np.pi) * lam**(-0.75) * np.exp(sl)


# ============================================================
# Computation
# ============================================================

results = {
    "metadata": {
        "script": "sec6_wilson_loop.py",
        "section": "6.2",
        "description": "1/2-BPS circular Wilson loop VEV in N=4 SYM conformal vacuum",
        "formulas": {
            "finite_N": "<W>_{U(N)} = (1/N) L^1_{N-1}(-lam/(4N)) exp(lam/(8N))",
            "planar": "<W>_{planar} = (2/sqrt(lam)) I_1(sqrt(lam))",
        },
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    },
    "finite_N_vs_planar": [],
    "planar_scan": [],
    "weak_coupling_check": [],
    "strong_coupling_check": [],
    "positivity_analytic": {},
    "structural_analysis": {},
}

# --- (A) Finite-N vs planar comparison ---
for N in [10, 50, 100, 500, 1000]:
    for lam in [1.0, 10.0, 100.0]:
        w_N = wilson_finite_N(lam, N)
        w_planar = wilson_planar(lam)
        results["finite_N_vs_planar"].append({
            "N": N,
            "lambda": lam,
            "W_finite_N": float(w_N),
            "W_planar": float(w_planar),
            "relative_diff": float(abs(w_N - w_planar) / w_planar),
        })

# --- (B) Planar scan ---
for lam in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
    w = wilson_planar(lam)
    results["planar_scan"].append({
        "lambda": lam,
        "sqrt_lambda": float(np.sqrt(lam)),
        "W_vev": float(w),
        "nonzero": bool(w > 0),
    })

# --- (C) Weak coupling check ---
for lam in [0.001, 0.005, 0.01, 0.05, 0.1]:
    exact = wilson_planar(lam)
    approx = wilson_weak(lam)
    results["weak_coupling_check"].append({
        "lambda": lam,
        "exact": float(exact),
        "approx": float(approx),
        "relative_error": float(abs(exact - approx) / exact),
    })

# --- (C) Strong coupling check ---
for lam in [20.0, 50.0, 100.0, 200.0, 500.0]:
    exact = wilson_planar(lam)
    asympt = wilson_strong(lam)
    results["strong_coupling_check"].append({
        "lambda": lam,
        "exact": float(exact),
        "asymptotic": float(asympt),
        "ratio": float(exact / asympt),
    })

# --- (D) Positivity: analytic argument ---
results["positivity_analytic"] = {
    "statement": "I_1(z) > 0 for all z > 0 (standard property of modified Bessel function)",
    "implication": "lambda > 0 => sqrt(lambda) > 0 => I_1(sqrt(lambda)) > 0 => <W> > 0",
    "conclusion": "<W(C_circle)> > 0 for all lambda > 0, analytically proven",
}

# --- Structural analysis ---
results["structural_analysis"] = {
    "vanishing_theorem_applies": False,
    "reason": [
        "W(C) is non-local: depends on contour C, not a single point x",
        "W(C) is dimensionless: no scaling dimension Delta > 0",
        "Contour C provides geometric data that absorbs scaling",
        "Conformal symmetry maps circles to circles: <W> = f(lambda, N)",
    ],
    "classification": "probe-response (defect type), not intrinsic bias",
    "key_result": "<W(C)> > 0 for all lambda > 0 in the symmetric vacuum",
    "gauge_group_note": "Exact formula is standard for U(N); planar limit is identical for SU(N)",
    "euclidean_note": "Wilson loop defined on Euclidean R^4 via Wick rotation",
}

# ============================================================
# Output: JSON
# ============================================================

output_json = "d:/おはなしのアトリエ/50_メタ思考ログ/知的エンターテインメント/存在論的不完全性原理/03_results/sec6_wilson_loop.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ============================================================
# Output: Log
# ============================================================

output_log = "d:/おはなしのアトリエ/50_メタ思考ログ/知的エンターテインメント/存在論的不完全性原理/03_results/sec6_wilson_loop.log"
with open(output_log, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("Section 6.2: 1/2-BPS Circular Wilson Loop in N=4 SYM\n")
    f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
    f.write("=" * 70 + "\n\n")

    # Finite-N vs planar
    f.write("--- (A) Finite-N exact vs planar limit ---\n")
    f.write(f"{'N':>6s} | {'lambda':>8s} | {'W(finite N)':>15s} | {'W(planar)':>15s} | {'rel diff':>12s}\n")
    f.write("-" * 65 + "\n")
    for row in results["finite_N_vs_planar"]:
        f.write(f"{row['N']:6d} | {row['lambda']:8.1f} | {row['W_finite_N']:15.8f} | {row['W_planar']:15.8f} | {row['relative_diff']:12.2e}\n")

    # Planar scan
    f.write("\n--- (B) Planar scan ---\n")
    f.write(f"{'lambda':>10s} | {'sqrt(lam)':>10s} | {'<W(C)>':>15s} | nonzero?\n")
    f.write("-" * 50 + "\n")
    for row in results["planar_scan"]:
        f.write(f"{row['lambda']:10.2f} | {row['sqrt_lambda']:10.4f} | {row['W_vev']:15.8f} | {'YES' if row['nonzero'] else 'NO'}\n")

    # Weak coupling
    f.write("\n--- (C) Weak coupling: <W> ~ 1 + lam/8 + lam^2/192 ---\n")
    f.write(f"{'lambda':>10s} | {'exact':>15s} | {'approx':>15s} | {'rel_err':>12s}\n")
    f.write("-" * 58 + "\n")
    for row in results["weak_coupling_check"]:
        f.write(f"{row['lambda']:10.4f} | {row['exact']:15.10f} | {row['approx']:15.10f} | {row['relative_error']:12.2e}\n")

    # Strong coupling
    f.write("\n--- (C) Strong coupling: <W> ~ sqrt(2/pi) lam^{-3/4} exp(sqrt(lam)) ---\n")
    f.write(f"{'lambda':>10s} | {'exact':>15s} | {'asymptotic':>15s} | {'ratio':>10s}\n")
    f.write("-" * 55 + "\n")
    for row in results["strong_coupling_check"]:
        f.write(f"{row['lambda']:10.1f} | {row['exact']:15.6f} | {row['asymptotic']:15.6f} | {row['ratio']:10.6f}\n")

    # Positivity
    f.write("\n--- (D) Positivity (analytic) ---\n")
    for k, v in results["positivity_analytic"].items():
        f.write(f"  {k}: {v}\n")

    # Structural
    f.write("\n--- Structural analysis ---\n")
    f.write(f"Vanishing theorem applies: {results['structural_analysis']['vanishing_theorem_applies']}\n")
    f.write("Reasons:\n")
    for r in results["structural_analysis"]["reason"]:
        f.write(f"  - {r}\n")
    f.write(f"Classification: {results['structural_analysis']['classification']}\n")
    f.write(f"Gauge group: {results['structural_analysis']['gauge_group_note']}\n")
    f.write(f"Euclidean: {results['structural_analysis']['euclidean_note']}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("CONCLUSION: <W(C)> > 0 for all lambda > 0 (analytically + numerically).\n")
    f.write("The vanishing theorem of Sec 2 does NOT extend to defect operators.\n")
    f.write("=" * 70 + "\n")

print(f"JSON: {output_json}")
print(f"Log:  {output_log}")
print("Done.")
