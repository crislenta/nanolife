"""Contract tests for the statistical-significance harness.

Covers:
- Welch's two-sample t-test against canonical reference values (scipy).
- Cohen's d with pooled SD on small samples.
- Sweep-summary comparison: shared/unmatched scenarios, per-metric stats,
  significance flagging at p < 0.05.
- Graceful degradation: n=0/1, constant samples, missing keys.

The reference t and p values below were captured from
``scipy.stats.ttest_ind(a, b, equal_var=False)`` at write time and are
hard-coded here so this suite has no scipy dependency at runtime.
"""
from __future__ import annotations

import math

import pytest

from nanosim.metrics import (
    HEADLINE_METRICS,
    compare_sweeps,
    format_compare,
    welch_t_test,
)


# ──────────────────────────────────────────────────────────────────────
# welch_t_test — match scipy
# ──────────────────────────────────────────────────────────────────────

class TestWelchTTest:
    """Reference values from scipy.stats.ttest_ind(..., equal_var=False)."""

    def test_identical_samples_p_equals_one(self):
        r = welch_t_test([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
        assert r["t"] == 0.0
        assert r["mean_diff"] == 0.0
        assert r["p"] == pytest.approx(1.0, abs=1e-6)
        assert r["cohen_d"] == 0.0

    def test_shifted_by_one_matches_scipy(self):
        # scipy: t=-1.0, p=0.34659350708733416, df=8
        r = welch_t_test([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        assert r["t"] == pytest.approx(-1.0, abs=1e-3)
        assert r["df"] == pytest.approx(8.0, abs=1e-3)
        assert r["p"] == pytest.approx(0.346594, abs=5e-5)

    def test_clear_difference_is_significant(self):
        # scipy: t=-26.8438..., p~3e-9
        r = welch_t_test(
            [0.1, 0.15, 0.2, 0.18, 0.12],
            [0.8, 0.85, 0.9, 0.88, 0.82],
        )
        assert r["t"] == pytest.approx(-26.8438, abs=1e-3)
        assert r["p"] < 0.001
        # Cohen's d on this huge gap should be very large in magnitude.
        assert abs(r["cohen_d"]) > 5.0

    def test_n2_each_matches_scipy(self):
        # scipy: t=-2.82843, df=2, p=0.10557281
        r = welch_t_test([1.0, 2.0], [3.0, 4.0])
        assert r["t"] == pytest.approx(-2.8284, abs=1e-3)
        assert r["df"] == pytest.approx(2.0, abs=1e-3)
        assert r["p"] == pytest.approx(0.10557, abs=5e-5)

    def test_unequal_variances_welch_df_is_fractional(self):
        # scipy: t=-6.8227..., p=0.000356..., df ~ 6.79 (fractional)
        r = welch_t_test(
            [0.5, 0.6, 0.55, 0.62, 0.58, 0.51],
            [0.7, 0.72, 0.68, 0.71, 0.69, 0.73],
        )
        assert r["t"] == pytest.approx(-6.8227, abs=1e-3)
        assert r["p"] == pytest.approx(0.000356, abs=5e-5)
        # Welch correction must produce a non-integer df < pooled (10).
        assert 5.0 < r["df"] < 10.0

    def test_near_zero_difference_high_p(self):
        # scipy: t=0.0814, p=0.94113 — tiny difference, no significance
        r = welch_t_test([3.1, 3.2, 3.0], [3.05, 3.15, 3.1, 3.08])
        assert r["p"] == pytest.approx(0.941126, abs=5e-5)
        assert r["p"] > 0.05

    def test_t_statistic_sign_matches_mean_order(self):
        # mean_a < mean_b -> t negative
        neg = welch_t_test([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        assert neg["t"] < 0
        assert neg["mean_diff"] < 0
        # mean_a > mean_b -> t positive
        pos = welch_t_test([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
        assert pos["t"] > 0
        assert pos["mean_diff"] > 0


# ──────────────────────────────────────────────────────────────────────
# welch_t_test — graceful degradation
# ──────────────────────────────────────────────────────────────────────

class TestWelchTTestEdgeCases:

    def test_empty_a_returns_p_one(self):
        r = welch_t_test([], [1.0, 2.0])
        assert r["n_a"] == 0 and r["n_b"] == 2
        assert r["p"] == 1.0
        assert r["t"] == 0.0
        # No exception.

    def test_single_sample_falls_back_safely(self):
        r = welch_t_test([1.0], [2.0, 3.0])
        assert r["n_a"] == 1
        assert r["p"] == 1.0
        assert r["t"] == 0.0
        assert r["cohen_d"] == 0.0

    def test_both_constant_equal_returns_p_one(self):
        r = welch_t_test([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])
        assert r["mean_diff"] == 0.0
        assert r["p"] == 1.0
        assert r["t"] == 0.0

    def test_both_constant_different_returns_p_zero(self):
        # Both samples have zero variance but different means — degenerate
        # but well-defined: this IS a real difference, just no noise.
        r = welch_t_test([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        assert r["mean_diff"] == -1.0
        assert r["p"] == 0.0
        assert math.isinf(r["cohen_d"])

    def test_cohen_d_scales_with_separation(self):
        # Same means, different spread -> larger effect with smaller spread.
        tight = welch_t_test([0.0, 0.01, -0.01], [1.0, 1.01, 0.99])
        loose = welch_t_test([0.0, 1.0, -1.0], [1.0, 2.0, 0.0])
        assert abs(tight["cohen_d"]) > abs(loose["cohen_d"])


# ──────────────────────────────────────────────────────────────────────
# compare_sweeps — operates on sweep summary.json shape
# ──────────────────────────────────────────────────────────────────────

def _summary(per_run):
    return {"sweep_id": "test", "per_run": per_run}


def _row(scenario, seed, **metrics):
    out = {"_scenario": scenario, "_seed": seed}
    out.update(metrics)
    return out


class TestCompareSweeps:

    def test_identical_sweeps_no_significance(self):
        rows = [
            _row("nanothrones", 0, survival_rate=0.5, cooperation_index=0.3,
                 narrative_coherence=0.4, action_diversity=0.6, emergence_index=5),
            _row("nanothrones", 1, survival_rate=0.55, cooperation_index=0.32,
                 narrative_coherence=0.42, action_diversity=0.58, emergence_index=5),
            _row("nanothrones", 2, survival_rate=0.52, cooperation_index=0.31,
                 narrative_coherence=0.41, action_diversity=0.59, emergence_index=6),
        ]
        report = compare_sweeps(_summary(rows), _summary(rows))
        assert report["shared_scenarios"] == ["nanothrones"]
        assert report["unmatched_scenarios"] == []
        assert report["significant"] == []
        # Every per-metric p must be exactly 1 since the data is identical.
        for metric, stats in report["by_scenario"]["nanothrones"].items():
            assert stats["mean_diff"] == 0.0
            assert stats["p"] == pytest.approx(1.0, abs=1e-6)

    def test_clearly_different_sweeps_flag_significance(self):
        a_rows = [_row("nanothrones", s, survival_rate=0.10 + 0.01 * s,
                       cooperation_index=0.05, narrative_coherence=0.30,
                       action_diversity=0.40, emergence_index=2) for s in range(5)]
        b_rows = [_row("nanothrones", s, survival_rate=0.90 + 0.01 * s,
                       cooperation_index=0.85, narrative_coherence=0.30,
                       action_diversity=0.40, emergence_index=2) for s in range(5)]
        report = compare_sweeps(_summary(a_rows), _summary(b_rows),
                                label_a="left", label_b="right")
        sig_metrics = {item["metric"] for item in report["significant"]}
        assert "survival_rate" in sig_metrics
        assert "cooperation_index" in sig_metrics
        # Constant-and-equal metrics (narrative/action/emergence) must NOT flag.
        assert "narrative_coherence" not in sig_metrics
        # Survival's effect size is huge; sign reflects A < B.
        sr = report["by_scenario"]["nanothrones"]["survival_rate"]
        assert sr["mean_diff"] < 0
        assert sr["cohen_d"] < -2.0

    def test_unmatched_scenarios_listed(self):
        a = _summary([_row("nanothrones", 0, survival_rate=0.5)])
        b = _summary([_row("nanozombie", 0, survival_rate=0.5)])
        report = compare_sweeps(a, b)
        assert report["shared_scenarios"] == []
        assert set(report["unmatched_scenarios"]) == {"nanothrones", "nanozombie"}
        assert report["by_scenario"] == {}
        assert report["significant"] == []

    def test_partially_overlapping_scenarios(self):
        a = _summary([
            _row("nanothrones", 0, survival_rate=0.5),
            _row("nanothrones", 1, survival_rate=0.55),
            _row("nanozombie", 0, survival_rate=0.7),
        ])
        b = _summary([
            _row("nanothrones", 0, survival_rate=0.6),
            _row("nanothrones", 1, survival_rate=0.62),
            _row("nanoception", 0, survival_rate=0.9),
        ])
        report = compare_sweeps(a, b)
        assert report["shared_scenarios"] == ["nanothrones"]
        assert set(report["unmatched_scenarios"]) == {"nanozombie", "nanoception"}
        assert "survival_rate" in report["by_scenario"]["nanothrones"]

    def test_missing_metric_leaves_empty_sample(self):
        # A row missing 'cooperation_index' should not crash; that metric
        # gets n=0 for that row's contribution.
        a_rows = [_row("nanothrones", s, survival_rate=0.5) for s in range(3)]
        b_rows = [_row("nanothrones", s, survival_rate=0.6) for s in range(3)]
        report = compare_sweeps(_summary(a_rows), _summary(b_rows))
        coop = report["by_scenario"]["nanothrones"]["cooperation_index"]
        assert coop["n_a"] == 0 and coop["n_b"] == 0
        assert coop["p"] == 1.0
        # The metric that IS present still works.
        assert "survival_rate" in report["by_scenario"]["nanothrones"]

    def test_default_metrics_are_headline(self):
        # Sanity: a freshly compared sweep tracks every headline metric.
        rows = [_row("nanothrones", s, survival_rate=0.5, cooperation_index=0.3,
                     narrative_coherence=0.4, action_diversity=0.6,
                     emergence_index=5) for s in range(3)]
        report = compare_sweeps(_summary(rows), _summary(rows))
        for metric in HEADLINE_METRICS:
            assert metric in report["by_scenario"]["nanothrones"]

    def test_format_compare_renders_significance_marker(self):
        a_rows = [_row("nanothrones", s, survival_rate=0.1) for s in range(5)]
        b_rows = [_row("nanothrones", s, survival_rate=0.9) for s in range(5)]
        report = compare_sweeps(_summary(a_rows), _summary(b_rows),
                                label_a="ctrl", label_b="exp")
        text = format_compare(report)
        assert "ctrl" in text and "exp" in text
        # The asterisk marks statistically significant rows in the table.
        assert " *" in text
        assert "significant differences" in text

    def test_format_compare_handles_no_overlap(self):
        a = _summary([_row("nanothrones", 0, survival_rate=0.5)])
        b = _summary([_row("nanozombie", 0, survival_rate=0.5)])
        text = format_compare(compare_sweeps(a, b))
        assert "no shared scenarios" in text


# ──────────────────────────────────────────────────────────────────────
# Mutation guardrails: a bug in t-test math should make tests fail.
# (Documented here so future mutations don't silently pass.)
# ──────────────────────────────────────────────────────────────────────

class TestMutationGuards:
    """Sanity checks that pin down arithmetic — meant to fail loudly if
    someone swaps Welch for Student or drops the (n-1) Bessel correction."""

    def test_uses_bessel_correction_not_population_var(self):
        # If the code used population variance (divide by n) instead of
        # sample variance (n-1), t for [1,2,3] vs [4,5,6] would be larger.
        # scipy Welch: t = -3.6742..., p=0.0211...
        r = welch_t_test([1, 2, 3], [4, 5, 6])
        assert r["t"] == pytest.approx(-3.6742, abs=1e-3)
        assert r["p"] == pytest.approx(0.0211, abs=5e-4)

    def test_two_sided_not_one_sided(self):
        # If the code reported a one-sided p-value, the value below would
        # be ~0.173 instead of ~0.346.
        r = welch_t_test([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        assert r["p"] == pytest.approx(0.346594, abs=5e-5)
        assert r["p"] > 0.2  # Definitely not one-sided.
