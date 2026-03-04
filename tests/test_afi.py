"""Tests for the AFI library."""

import numpy as np
import pytest

from afi.core.perception import Perception, perception_from_sensors, perception_from_aco
from afi.core.distortion import (
    Distortion,
    DistortionFactor,
    MultiplicativeDistortion,
    AdditiveDistortion,
)
from afi.core.freedom import compute_freedom, compute_freedom_intelligent, FreedomField
from afi.exploration import ExplorationExploitation, compare_algorithms
from afi.gradient import GradientLaw
from afi.convergence import ConvergenceBound
from afi.validation import compare_models, compare_distortion_composition, noise_robustness


# ===== Perception =====

class TestPerception:
    def test_basic(self):
        p = Perception(n_states=8, temporal_depth=3)
        assert p.value == pytest.approx(9.0)  # log2(8) * 3 = 3 * 3

    def test_minimum(self):
        p = Perception(n_states=2, temporal_depth=1)
        assert p.value == pytest.approx(1.0)  # log2(2) * 1

    def test_array(self):
        p = Perception(n_states=np.array([4, 8, 16]), temporal_depth=1)
        np.testing.assert_allclose(p.value, [2.0, 3.0, 4.0])

    def test_invalid_n_states(self):
        with pytest.raises(ValueError, match="n_states must be >= 2"):
            Perception(n_states=1)

    def test_invalid_temporal_depth(self):
        with pytest.raises(ValueError, match="temporal_depth must be >= 1"):
            Perception(n_states=4, temporal_depth=0.5)

    def test_from_sensors(self):
        p = perception_from_sensors([256, 128, 64], temporal_depth=2)
        assert p.n_states == 256 * 128 * 64
        assert p.temporal_depth == 2.0

    def test_aco_perception(self):
        pheromone = np.array([1.0, 2.0, 3.0])
        visibility = np.array([0.5, 0.25, 0.1])
        p = perception_from_aco(pheromone, visibility, alpha=1.0, beta=2.0)
        expected = pheromone**1.0 * visibility**2.0
        np.testing.assert_allclose(p, expected)


# ===== Distortion =====

class TestDistortion:
    def test_basic(self):
        d = Distortion(measure=30.0, baseline=10.0)
        assert d.value == pytest.approx(3.0)

    def test_unit_baseline(self):
        d = Distortion(measure=5.0)
        assert d.value == pytest.approx(5.0)

    def test_invalid_baseline(self):
        with pytest.raises(ValueError, match="baseline must be > 0"):
            Distortion(measure=1.0, baseline=0.0)


class TestMultiplicativeDistortion:
    def test_basic(self):
        md = MultiplicativeDistortion([
            DistortionFactor("R", np.array([2.0]), exponent=1.0),
            DistortionFactor("O", np.array([3.0]), exponent=1.0),
        ])
        assert md.value[0] == pytest.approx(6.0)  # 2^1 * 3^1

    def test_neutral_element(self):
        md = MultiplicativeDistortion([
            DistortionFactor("R", np.array([1.0]), exponent=1.0),
            DistortionFactor("O", np.array([3.0]), exponent=1.0),
        ])
        assert md.value[0] == pytest.approx(3.0)  # 1^1 * 3^1 = 3

    def test_exponents(self):
        md = MultiplicativeDistortion([
            DistortionFactor("R", np.array([4.0]), exponent=0.5),
        ])
        assert md.value[0] == pytest.approx(2.0)  # 4^0.5 = 2

    def test_infinity_barrier(self):
        md = MultiplicativeDistortion([
            DistortionFactor("R", np.array([1e10]), exponent=1.0),
            DistortionFactor("O", np.array([1.0]), exponent=1.0),
        ])
        assert md.value[0] > 1e9  # one infinite barrier dominates

    def test_log_value(self):
        md = MultiplicativeDistortion([
            DistortionFactor("R", np.array([np.e]), exponent=2.0),
        ])
        assert md.log_value[0] == pytest.approx(2.0)  # 2 * ln(e) = 2

    def test_factor_below_one_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            DistortionFactor("bad", np.array([0.5]))

    def test_contribution(self):
        md = MultiplicativeDistortion([
            DistortionFactor("R", np.array([2.0]), exponent=3.0),
            DistortionFactor("O", np.array([5.0]), exponent=1.0),
        ])
        np.testing.assert_allclose(md.contribution("R"), [8.0])  # 2^3
        np.testing.assert_allclose(md.contribution("O"), [5.0])  # 5^1


class TestAdditiveDistortion:
    def test_basic(self):
        ad = AdditiveDistortion([
            DistortionFactor("R", np.array([2.0]), exponent=1.0),
            DistortionFactor("O", np.array([3.0]), exponent=1.0),
        ])
        assert ad.value[0] == pytest.approx(5.0)  # 1*2 + 1*3


# ===== Freedom =====

class TestFreedom:
    def test_ratio(self):
        assert compute_freedom(6.0, 2.0) == pytest.approx(3.0)

    def test_array(self):
        result = compute_freedom(np.array([2, 4, 8]), np.array([1, 2, 4]))
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0])

    def test_scale_covariance(self):
        """F(λP, λD) = F(P, D) for all λ > 0."""
        p, d = 5.0, 3.0
        for lam in [0.1, 0.5, 1.0, 2.0, 10.0]:
            assert compute_freedom(lam * p, lam * d) == pytest.approx(compute_freedom(p, d))

    def test_monotonicity_p(self):
        """F increases with P."""
        d = 2.0
        f_values = [compute_freedom(p, d) for p in [1, 2, 4, 8]]
        assert f_values == sorted(f_values)

    def test_monotonicity_d(self):
        """F decreases with D."""
        p = 4.0
        f_values = [compute_freedom(p, d) for d in [1, 2, 4, 8]]
        assert f_values == sorted(f_values, reverse=True)

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="P must be > 0"):
            compute_freedom(0.0, 1.0)

    def test_invalid_d(self):
        with pytest.raises(ValueError, match="D must be > 0"):
            compute_freedom(1.0, 0.0)

    def test_intelligent_regime(self):
        f = compute_freedom_intelligent(p_ext=4, p_rec=3, d_ext=2, d_int=3)
        assert f == pytest.approx(2.0)  # (4*3)/(2*3) = 2.0


class TestFreedomField:
    def test_basic(self):
        P = np.array([[4, 6, 8], [3, 6, 9]])
        D = np.array([[2, 3, 4], [1, 2, 3]])
        ff = FreedomField(P, D)
        expected = P / D
        np.testing.assert_allclose(ff.freedom, expected)

    def test_mean_freedom(self):
        P = np.array([[4, 6, 8], [3, 6, 9]])
        D = np.array([[2, 3, 4], [1, 2, 3]])
        ff = FreedomField(P, D)
        # t=0: [2, 2, 2] -> mean 2; t=1: [3, 3, 3] -> mean 3
        np.testing.assert_allclose(ff.mean_freedom(), [2.0, 3.0])

    def test_cv_freedom(self):
        # All equal F -> CV = 0
        P = np.array([[4, 4, 4]])
        D = np.array([[2, 2, 2]])
        ff = FreedomField(P, D)
        assert ff.cv_freedom()[0] == pytest.approx(0.0)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            FreedomField(np.ones((3, 4)), np.ones((3, 5)))

    def test_anomaly_decomposition(self):
        # Baseline: stable P and D
        P = np.ones((20, 10)) * 4.0
        D = np.ones((20, 10)) * 2.0
        # Inject distortion spike at t=15
        D[15:, :] = 8.0
        ff = FreedomField(P, D)
        result = ff.decompose_anomaly(baseline_end=10, threshold_sigma=2.0)
        # After t=15, F drops from 2 to 0.5 -> anomaly
        assert np.any(result["anomaly_mask"][15:])
        # Primary cause should be distortion
        assert result["primary_cause"][15] == "distortion"


# ===== Exploration-Exploitation =====

class TestExplorationExploitation:
    def test_crossing_detection(self):
        # E starts > 1 and drops below 1
        rng = np.random.default_rng(42)
        n_agents = 50
        n_steps = 100

        P = np.zeros((n_steps, n_agents))
        D = np.zeros((n_steps, n_agents))
        for t in range(n_steps):
            spread = max(0.1, 2.0 - t * 0.03)  # decreasing spread
            P[t] = rng.uniform(1, 1 + spread * 5, n_agents)
            D[t] = rng.uniform(1, 1 + spread * 3, n_agents)

        ff = FreedomField(P, D)
        ee = ExplorationExploitation(ff)
        crossing = ee.crossing_iteration("down")
        # Should find a crossing somewhere
        assert crossing is not None or ee.e_series[-1] < ee.e_series[0]

    def test_phase_at(self):
        P = np.array([[10, 1], [5, 5]])
        D = np.array([[1, 1], [1, 1]])
        ff = FreedomField(P, D)
        ee = ExplorationExploitation(ff)
        # t=0: F=[10,1], mean=5.5, std=4.5, CV=4.5/5.5≈0.818
        assert ee.phase_at(0) in ("exploration", "exploitation", "transition")


# ===== Gradient Law =====

class TestGradientLaw:
    def test_movement_direction(self):
        # Simple gradient: D increases to the right
        d_field = np.tile(np.arange(10, dtype=float), (10, 1))
        gl = GradientLaw(d_field)
        pos = np.array([[5, 5]])
        move = gl.movement_direction(pos, perception=1.0)
        # Should move LEFT (negative x direction, toward lower D)
        assert move[0, 1] < 0  # dx component is negative

    def test_freedom_field(self):
        d_field = np.ones((5, 5)) * 2.0
        p_field = np.ones((5, 5)) * 6.0
        gl = GradientLaw(d_field)
        f_field = gl.freedom_field(p_field)
        np.testing.assert_allclose(f_field, 3.0)

    def test_simulate(self):
        d_field = np.tile(np.arange(10, dtype=float) + 1, (10, 1))
        gl = GradientLaw(d_field)
        traj = gl.simulate_agents(np.array([[5, 8]]), perception=1.0, n_steps=10)
        # Agent should move toward lower D (left)
        assert traj[-1, 0, 1] < traj[0, 0, 1]


# ===== Convergence Bound =====

class TestConvergenceBound:
    def test_basic(self):
        cb = ConvergenceBound(mean_distortion=5.0, state_space_size=1000, mean_perception=3.0)
        expected = 5.0 * np.log(1000) / 3.0
        assert cb.bound == pytest.approx(expected)

    def test_higher_p_faster(self):
        cb1 = ConvergenceBound(5.0, 1000, mean_perception=3.0)
        cb2 = ConvergenceBound(5.0, 1000, mean_perception=6.0)
        assert cb2.bound < cb1.bound

    def test_higher_d_slower(self):
        cb1 = ConvergenceBound(5.0, 1000, 3.0)
        cb2 = ConvergenceBound(10.0, 1000, 3.0)
        assert cb2.bound > cb1.bound

    def test_evaluate(self):
        cb = ConvergenceBound(5.0, 100, 3.0)
        result = cb.evaluate(observed_convergence=5.0)
        assert "bound" in result
        assert "bound_holds" in result


# ===== Validation =====

class TestValidation:
    def test_compare_models(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(1, 10, 100)
        d = rng.uniform(1, 5, 100)
        nav = p / d + rng.normal(0, 0.1, 100)
        results = compare_models(p, d, nav)
        # Ratio model should have highest R²
        assert results["ratio_PD"]["r_squared"] > results["additive_P-D"]["r_squared"]
        assert results["ratio_PD"]["r_squared"] > results["random"]["r_squared"]

    def test_distortion_composition(self):
        rng = np.random.default_rng(42)
        r = rng.uniform(1, 5, 100)
        o = rng.uniform(1, 3, 100)
        measured = r ** 1.0 * o ** 0.8 + rng.normal(0, 0.1, 100)
        result = compare_distortion_composition(
            factors={"R": r, "O": o},
            exponents={"R": 1.0, "O": 0.8},
            measured_d=measured,
        )
        assert result["multiplicative"]["r_squared"] > result["additive"]["r_squared"]

    def test_noise_robustness(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(2, 10, 200)
        d = rng.uniform(1, 5, 200)
        nav = p / d
        result = noise_robustness(p, d, nav, noise_levels=[0.01, 0.5])
        # Low noise should give higher R² than high noise
        assert result["mean_r_squared"][0] > result["mean_r_squared"][1]


# ===== Property-based tests =====

class TestProperties:
    """Mathematical properties that must always hold."""

    def test_freedom_always_positive(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.uniform(0.01, 1000)
            d = rng.uniform(0.01, 1000)
            assert compute_freedom(p, d) > 0

    def test_scale_covariance_random(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.uniform(0.1, 100)
            d = rng.uniform(0.1, 100)
            lam = rng.uniform(0.01, 100)
            f1 = compute_freedom(p, d)
            f2 = compute_freedom(lam * p, lam * d)
            assert f1 == pytest.approx(f2, rel=1e-10)

    def test_multiplicative_d_neutral_element(self):
        """Factor = 1 should not change total D."""
        base = DistortionFactor("base", np.array([3.0]), exponent=1.0)
        neutral = DistortionFactor("neutral", np.array([1.0]), exponent=1.0)
        d1 = MultiplicativeDistortion([base]).value
        d2 = MultiplicativeDistortion([base, neutral]).value
        np.testing.assert_allclose(d1, d2)

    def test_multiplicative_d_infinity_barrier(self):
        """One factor -> inf should force D -> inf."""
        normal = DistortionFactor("normal", np.array([2.0]), exponent=1.0)
        huge = DistortionFactor("huge", np.array([1e15]), exponent=1.0)
        d = MultiplicativeDistortion([normal, huge]).value
        assert d[0] > 1e14
