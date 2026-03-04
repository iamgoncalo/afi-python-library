#!/usr/bin/env python3
"""Run all AFI tests without pytest dependency."""
import sys, traceback
import numpy as np

passed = failed = 0
errors = []

def approx(a, b, rel=1e-6):
    return abs(a - b) < max(abs(a), abs(b), 1e-12) * rel + 1e-12

def test(name, func):
    global passed, failed
    try:
        func()
        passed += 1
        print(f"  ✓ {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: {e}")

print("=" * 60)
print("AFI TEST SUITE")
print("=" * 60)

# ===== PERCEPTION =====
print("\nPerception:")
from afi.core.perception import Perception, perception_from_sensors, perception_from_aco

def t_p_basic():
    p = Perception(n_states=8, temporal_depth=3)
    if not approx(p.value, 9.0): raise ValueError(f"Expected 9.0, got {p.value}")
test("P = log2(8)*3 = 9.0", t_p_basic)

def t_p_min():
    p = Perception(n_states=2, temporal_depth=1)
    if not approx(p.value, 1.0): raise ValueError(f"Expected 1.0, got {p.value}")
test("P = log2(2)*1 = 1.0", t_p_min)

def t_p_arr():
    p = Perception(n_states=np.array([4,8,16]), temporal_depth=1)
    np.testing.assert_allclose(p.value, [2.0, 3.0, 4.0])
test("array perception", t_p_arr)

def t_p_invalid():
    try:
        Perception(n_states=1)
        raise RuntimeError("Should have raised ValueError")
    except ValueError:
        pass
test("reject n_states < 2", t_p_invalid)

def t_p_sensors():
    p = perception_from_sensors([256, 128, 64], temporal_depth=2)
    if p.n_states != 256*128*64: raise ValueError(f"Got {p.n_states}")
test("from_sensors", t_p_sensors)

def t_p_aco():
    r = perception_from_aco(np.array([1.0, 2.0]), np.array([0.5, 0.25]), alpha=1.0, beta=2.0)
    np.testing.assert_allclose(r, [1.0*0.25, 2.0*0.0625])
test("ACO perception", t_p_aco)

# ===== DISTORTION =====
print("\nDistortion:")
from afi.core.distortion import Distortion, DistortionFactor, MultiplicativeDistortion, AdditiveDistortion

def t_d_basic():
    if not approx(Distortion(30.0, 10.0).value, 3.0): raise ValueError()
test("D = 30/10 = 3.0", t_d_basic)

def t_d_invalid():
    try:
        Distortion(1.0, baseline=0.0)
        raise RuntimeError("Should have raised")
    except ValueError:
        pass
test("reject baseline=0", t_d_invalid)

def t_md_basic():
    md = MultiplicativeDistortion([
        DistortionFactor("R", np.array([2.0]), 1.0),
        DistortionFactor("O", np.array([3.0]), 1.0),
    ])
    if not approx(md.value[0], 6.0): raise ValueError(f"Expected 6.0, got {md.value[0]}")
test("mult D: 2^1 * 3^1 = 6", t_md_basic)

def t_md_neutral():
    md = MultiplicativeDistortion([
        DistortionFactor("R", np.array([1.0]), 1.0),
        DistortionFactor("O", np.array([3.0]), 1.0),
    ])
    if not approx(md.value[0], 3.0): raise ValueError(f"Expected 3.0, got {md.value[0]}")
test("mult D neutral: 1^1 * 3^1 = 3", t_md_neutral)

def t_md_exp():
    md = MultiplicativeDistortion([DistortionFactor("R", np.array([4.0]), 0.5)])
    if not approx(md.value[0], 2.0): raise ValueError(f"Expected 2.0, got {md.value[0]}")
test("mult D exponent: 4^0.5 = 2", t_md_exp)

def t_md_log():
    md = MultiplicativeDistortion([DistortionFactor("R", np.array([np.e]), 2.0)])
    if not approx(md.log_value[0], 2.0): raise ValueError(f"Expected 2.0, got {md.log_value[0]}")
test("mult D log: 2*ln(e) = 2", t_md_log)

def t_md_inf():
    md = MultiplicativeDistortion([
        DistortionFactor("R", np.array([1e10]), 1.0),
        DistortionFactor("O", np.array([1.0]), 1.0),
    ])
    if md.value[0] <= 1e9: raise ValueError(f"Should be huge, got {md.value[0]}")
test("infinity barrier dominates", t_md_inf)

def t_ad():
    ad = AdditiveDistortion([
        DistortionFactor("R", np.array([2.0]), 1.0),
        DistortionFactor("O", np.array([3.0]), 1.0),
    ])
    if not approx(ad.value[0], 5.0): raise ValueError(f"Expected 5.0, got {ad.value[0]}")
test("add D: 1*2 + 1*3 = 5", t_ad)

def t_factor_below():
    try:
        DistortionFactor("bad", np.array([0.5]))
        raise RuntimeError("Should have raised")
    except ValueError:
        pass
test("reject factor < 1", t_factor_below)

def t_md_contribution():
    md = MultiplicativeDistortion([
        DistortionFactor("R", np.array([2.0]), 3.0),
        DistortionFactor("O", np.array([5.0]), 1.0),
    ])
    np.testing.assert_allclose(md.contribution("R"), [8.0])  # 2^3
    np.testing.assert_allclose(md.contribution("O"), [5.0])
test("contribution: R^3=8, O^1=5", t_md_contribution)

# ===== FREEDOM =====
print("\nFreedom:")
from afi.core.freedom import compute_freedom, compute_freedom_intelligent, FreedomField

def t_f_basic():
    if not approx(compute_freedom(6.0, 2.0), 3.0): raise ValueError()
test("F = 6/2 = 3.0", t_f_basic)

def t_f_arr():
    np.testing.assert_allclose(compute_freedom(np.array([2,4,8]), np.array([1,2,4])), [2.,2.,2.])
test("array Freedom", t_f_arr)

def t_f_scale():
    for lam in [0.1, 0.5, 1.0, 2.0, 10.0]:
        f1 = compute_freedom(5.0, 3.0)
        f2 = compute_freedom(lam*5, lam*3)
        if not approx(f1, f2, rel=1e-10):
            raise ValueError(f"λ={lam}: {f1} != {f2}")
test("scale covariance F(λP,λD) = F(P,D)", t_f_scale)

def t_f_mono_p():
    fs = [compute_freedom(p, 2.0) for p in [1,2,4,8]]
    if fs != sorted(fs): raise ValueError(f"Not monotonic: {fs}")
test("monotonicity in P", t_f_mono_p)

def t_f_mono_d():
    fs = [compute_freedom(4.0, d) for d in [1,2,4,8]]
    if fs != sorted(fs, reverse=True): raise ValueError(f"Not decreasing: {fs}")
test("monotonicity in D (decreasing)", t_f_mono_d)

def t_f_invalid_p():
    try:
        compute_freedom(0.0, 1.0)
        raise RuntimeError("Should have raised")
    except ValueError:
        pass
test("reject P=0", t_f_invalid_p)

def t_f_intelligent():
    f = compute_freedom_intelligent(4, 3, 2, 3)
    if not approx(f, 2.0): raise ValueError(f"Expected 2.0, got {f}")
test("intelligent: (4*3)/(2*3) = 2.0", t_f_intelligent)

def t_ff_mean():
    P = np.array([[4,6,8],[3,6,9]], dtype=float)
    D = np.array([[2,3,4],[1,2,3]], dtype=float)
    ff = FreedomField(P, D)
    np.testing.assert_allclose(ff.mean_freedom(), [2.0, 3.0])
test("FreedomField mean", t_ff_mean)

def t_ff_cv_zero():
    ff = FreedomField(np.array([[4.,4.,4.]]), np.array([[2.,2.,2.]]))
    if not approx(ff.cv_freedom()[0], 0.0): raise ValueError()
test("FreedomField CV=0 for uniform", t_ff_cv_zero)

def t_ff_anomaly():
    P = np.ones((20,10)) * 4.0
    D = np.ones((20,10)) * 2.0
    D[15:, :] = 8.0
    ff = FreedomField(P, D)
    r = ff.decompose_anomaly(baseline_end=10, threshold_sigma=2.0)
    if not np.any(r["anomaly_mask"][15:]): raise ValueError("No anomaly detected")
    if r["primary_cause"][15] != "distortion": raise ValueError(f"Wrong cause: {r['primary_cause'][15]}")
test("anomaly decomposition → distortion", t_ff_anomaly)

# ===== EXPLORATION =====
print("\nExploration-Exploitation:")
from afi.exploration import ExplorationExploitation

def t_ee_phase():
    P = np.array([[10.,1.],[5.,5.]])
    D = np.array([[1.,1.],[1.,1.]])
    ff = FreedomField(P, D)
    ee = ExplorationExploitation(ff)
    phase = ee.phase_at(0)
    if phase not in ("exploration", "exploitation", "transition"):
        raise ValueError(f"Unexpected phase: {phase}")
test("phase detection works", t_ee_phase)

def t_ee_summary():
    rng = np.random.default_rng(42)
    P = np.abs(rng.normal(5, 2, (50, 20))) + 0.1
    D = np.abs(rng.normal(3, 1, (50, 20))) + 0.1
    ff = FreedomField(P, D)
    ee = ExplorationExploitation(ff)
    s = ee.summary()
    if "crossing_iteration" not in s: raise ValueError("Missing key")
    if "phase_durations" not in s: raise ValueError("Missing key")
test("summary dict complete", t_ee_summary)

# ===== GRADIENT =====
print("\nGradient Law:")
from afi.gradient import GradientLaw

def t_grad_direction():
    d_field = np.tile(np.arange(10, dtype=float)+1, (10,1))
    gl = GradientLaw(d_field)
    move = gl.movement_direction(np.array([[5,5]]), perception=1.0)
    if move[0,1] >= 0: raise ValueError(f"Should move left, dx={move[0,1]}")
test("move toward lower D", t_grad_direction)

def t_grad_ff():
    gl = GradientLaw(np.ones((5,5))*2.0)
    f = gl.freedom_field(np.ones((5,5))*6.0)
    np.testing.assert_allclose(f, 3.0)
test("F field = P/D on grid", t_grad_ff)

def t_grad_sim():
    gl = GradientLaw(np.tile(np.arange(10, dtype=float)+1, (10,1)))
    traj = gl.simulate_agents(np.array([[5,8]]), perception=1.0, n_steps=10)
    if traj[-1,0,1] >= traj[0,0,1]: raise ValueError("Should move left")
test("simulate → agent moves left", t_grad_sim)

# ===== CONVERGENCE =====
print("\nConvergence Bound:")
from afi.convergence import ConvergenceBound

def t_cb():
    cb = ConvergenceBound(5.0, 1000, 3.0)
    expected = 5.0 * np.log(1000) / 3.0
    if not approx(cb.bound, expected): raise ValueError(f"{cb.bound} != {expected}")
test("bound = D̄·ln(|S|)/P_mean", t_cb)

def t_cb_p():
    if ConvergenceBound(5,1000,6.0).bound >= ConvergenceBound(5,1000,3.0).bound:
        raise ValueError("Higher P should give tighter bound")
test("higher P → faster convergence", t_cb_p)

def t_cb_d():
    if ConvergenceBound(10,1000,3).bound <= ConvergenceBound(5,1000,3).bound:
        raise ValueError("Higher D should give looser bound")
test("higher D → slower convergence", t_cb_d)

# ===== VALIDATION =====
print("\nValidation:")
from afi.validation import compare_models, compare_distortion_composition, noise_robustness

def t_val_ratio():
    rng = np.random.default_rng(42)
    p = rng.uniform(1,10,100)
    d = rng.uniform(1,5,100)
    nav = p/d + rng.normal(0, 0.1, 100)
    r = compare_models(p, d, nav)
    if r["ratio_PD"]["r_squared"] <= r["additive_P-D"]["r_squared"]:
        raise ValueError("Ratio should beat additive")
    if r["ratio_PD"]["r_squared"] <= r["random"]["r_squared"]:
        raise ValueError("Ratio should beat random")
test("ratio > additive > random", t_val_ratio)

def t_val_mult():
    rng = np.random.default_rng(42)
    r = rng.uniform(1,5,100)
    o = rng.uniform(1,3,100)
    measured = r**1.0 * o**0.8 + rng.normal(0, 0.1, 100)
    result = compare_distortion_composition({"R": r, "O": o}, {"R": 1.0, "O": 0.8}, measured)
    if result["multiplicative"]["r_squared"] <= result["additive"]["r_squared"]:
        raise ValueError("Mult should beat additive")
test("mult D > additive D", t_val_mult)

def t_val_noise():
    rng = np.random.default_rng(42)
    p = rng.uniform(2,10,200)
    d = rng.uniform(1,5,200)
    r = noise_robustness(p, d, p/d, noise_levels=[0.01, 0.5])
    if r["mean_r_squared"][0] <= r["mean_r_squared"][1]:
        raise ValueError("Low noise should give higher R²")
test("low noise > high noise", t_val_noise)

# ===== PROPERTIES =====
print("\nMathematical Properties:")

def t_prop_positive():
    rng = np.random.default_rng(42)
    for _ in range(100):
        f = compute_freedom(rng.uniform(0.01,1000), rng.uniform(0.01,1000))
        if f <= 0: raise ValueError(f"F={f} not positive")
test("F always positive (100 random)", t_prop_positive)

def t_prop_scale():
    rng = np.random.default_rng(42)
    for _ in range(100):
        p, d = rng.uniform(0.1,100), rng.uniform(0.1,100)
        lam = rng.uniform(0.01,100)
        if not approx(compute_freedom(p,d), compute_freedom(lam*p, lam*d), rel=1e-10):
            raise ValueError("Scale covariance violated")
test("scale covariance (100 random)", t_prop_scale)

def t_prop_neutral():
    base = DistortionFactor("b", np.array([3.0]), 1.0)
    neutral = DistortionFactor("n", np.array([1.0]), 1.0)
    np.testing.assert_allclose(
        MultiplicativeDistortion([base]).value,
        MultiplicativeDistortion([base, neutral]).value)
test("mult D neutral element", t_prop_neutral)

# ===== SUMMARY =====
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*60}")
if errors:
    print("\nFAILURES:")
    for name, msg in errors:
        print(f"  {name}: {msg}")
sys.exit(0 if failed == 0 else 1)
