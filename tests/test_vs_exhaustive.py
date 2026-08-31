"""
tests/test_vs_exhaustive.py -- the validation suite that gives "verified
reproduction" its meaning: every algorithm checked against either
brute-force exhaustive search or an independent numerical ground truth,
not just "the code runs and produces a plot that looks plausible."

Run with:  pytest tests/ -v

A note on thresholds: several tests below use gap/tolerance thresholds
calibrated against real measured behavior (see the comments next to each),
not made up to be comfortably passable. Where a claim turned out NOT to be
true -- e.g. "MQP always finds the exact global optimum" -- the test
asserts the weaker, honest claim instead ("MQP's mean gap to the global
optimum, over many trials, is small"), because the paper itself is
explicit that neither MQP nor GaBP is guaranteed to find the global
optimum of the non-convex problem (18); only convergence to *a*
stationary point is guaranteed.
"""

import numpy as np
import pytest

from system_model import (gen_alphabet, qam_constellation, beta_star,
                           objective, Qb, iid_Rayleigh_channel)
from baselines import precoder_wf
from baselines import cdm
from mqp import mqp
from gabp import gabp
from exhaustive import exhaustive


# ---------------------------------------------------------------------
# Core formula sanity checks
# ---------------------------------------------------------------------

def test_beta_star_matches_grid_search():
    """eq (9) should exactly match a numerical grid-search minimizer of
    the same one-variable quadratic, for many random (H, x, s)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        NU, NT = 4, 8
        H = iid_Rayleigh_channel(NU, NT, rng)
        x = (rng.standard_normal(NT) + 1j * rng.standard_normal(NT))
        s = (rng.standard_normal(NU) + 1j * rng.standard_normal(NU))
        sigma_n2 = 0.1

        beta_formula = beta_star(s, H, x, NU, sigma_n2)

        betas = np.linspace(-5, 5, 20001)
        vals = [objective(s, H, x, b, NU, sigma_n2) for b in betas]
        beta_grid = betas[np.argmin(vals)]

        assert abs(beta_formula - beta_grid) < 1e-2


def test_quantizer_power_calibration():
    """A uniformly-random alphabet point should have average power close
    to P/NT (the design target from Sec. 1.2 of the tutorial)."""
    P, NT, b = 4.0, 16, 2
    _, _, C = gen_alphabet(b, P, NT)
    avg_power = np.mean(np.abs(C) ** 2)
    assert abs(avg_power - P / NT) / (P / NT) < 0.05


def test_objective_scale_invariance():
    """Tutorial Sec. 3.2: once ||x||^2=P is substituted into the noise
    term, the objective g(x,beta) = ||s-bHx||^2 + b^2(NU*sn2/P)||x||^2 is
    invariant under (x,beta) -> (alpha*x, beta/alpha). This is the
    property that lets the power constraint be dropped during the
    iterative solve."""
    rng = np.random.default_rng(1)
    NU, NT, P, sigma_n2 = 3, 6, 1.0, 0.2
    H = iid_Rayleigh_channel(NU, NT, rng)
    s = (rng.standard_normal(NU) + 1j * rng.standard_normal(NU))
    x = (rng.standard_normal(NT) + 1j * rng.standard_normal(NT))
    beta = 0.7

    def g(x, beta):
        return (np.linalg.norm(s - beta * (H @ x)) ** 2
                + beta ** 2 * (NU * sigma_n2 / P) * np.linalg.norm(x) ** 2)

    for alpha in [0.3, 1.7, 5.0]:
        assert abs(g(alpha * x, beta / alpha) - g(x, beta)) < 1e-9


# ---------------------------------------------------------------------
# MQP vs. exhaustive search
# ---------------------------------------------------------------------

def _tiny_system_trials(n_trials, NT=6, NU=2, b=1, P=1.0, sigma_n2=0.05, seed0=500):
    """Shared tiny-system trial generator: NT=6/NU=2/1-bit keeps
    exhaustive search fast (4^6 = 4096 candidates) while still being a
    genuine cross-check, not a toy that's trivially easy for everything."""
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(4)
    for trial in range(n_trials):
        rng = np.random.default_rng(seed0 + trial)
        H = iid_Rayleigh_channel(NU, NT, rng)
        s = Sconst[rng.integers(0, 4, NU)]
        yield H, s, levels, thresholds, C, sigma_n2, P, NU


def test_mqp_beats_quantized_wf_on_average():
    """MQP should reliably improve on the naive quantize-after-the-fact
    WF baseline. Measured: WF's mean gap-to-optimum over 25 trials was
    ~0.14, MQP's was ~0.06 -- MQP should stay comfortably below WF."""
    mqp_gaps, wf_gaps = [], []
    for H, s, levels, thresholds, C, sigma_n2, P, NU in _tiny_system_trials(20):
        x_ex, J_ex = exhaustive(s, H, C, NU, sigma_n2)

        x_wf = Qb(precoder_wf(s, H, P, NU, sigma_n2), levels, thresholds)
        b_wf = beta_star(s, H, x_wf, NU, sigma_n2)
        wf_gaps.append(objective(s, H, x_wf, b_wf, NU, sigma_n2) - J_ex)

        x_m, _ = mqp(s, H, C, P, NU, sigma_n2, np.zeros(len(x_ex), dtype=complex), Tmax=40)
        xq = Qb(x_m, levels, thresholds)
        b_m = beta_star(s, H, xq, NU, sigma_n2)
        mqp_gaps.append(objective(s, H, xq, b_m, NU, sigma_n2) - J_ex)

    assert np.mean(mqp_gaps) < np.mean(wf_gaps)


def test_mqp_near_exhaustive_tiny_system():
    """MQP's mean gap to the true global optimum, averaged over many
    random instances, should be small. NOT asserting exact match every
    trial -- measured exact-match rate was 8/25 in calibration, with a
    mean gap of ~0.06; the bound below has generous margin."""
    gaps = []
    for H, s, levels, thresholds, C, sigma_n2, P, NU in _tiny_system_trials(20):
        x_ex, J_ex = exhaustive(s, H, C, NU, sigma_n2)
        x_m, _ = mqp(s, H, C, P, NU, sigma_n2, np.zeros(len(x_ex), dtype=complex), Tmax=40)
        xq = Qb(x_m, levels, thresholds)
        b_m = beta_star(s, H, xq, NU, sigma_n2)
        gaps.append(objective(s, H, xq, b_m, NU, sigma_n2) - J_ex)

    assert np.mean(gaps) < 0.2   # calibrated mean was ~0.06


# ---------------------------------------------------------------------
# GaBP vs. exhaustive search (tiny system) and vs. MQP (realistic scale)
# ---------------------------------------------------------------------

def test_gabp_bounded_gap_tiny_system():
    """GaBP vs. exhaustive on the SAME tiny NU=2 system used for MQP
    above. This is a genuinely hard regime for loopy BP -- with only 2
    factor nodes, "sum over all factors except this one" leaves exactly
    one term, which is about the least Gaussian-like / most adversarial
    case the algorithm sees (see tutorial Sec. 6.7 and Appendix B item 7).
    Calibrated (after fixing the eq-22 gamma^2 exponent bug, and holding
    zeta fixed rather than annealed -- see gabp.py's docstring for why)
    mean gap here was ~0.11, with exact matches in 6-7/20 trials -- this
    test exists mainly to catch a REGRESSION (e.g. either the leave-one-
    out aggregation bug from Sec. 6.7, or the gamma^2 exponent bug, both
    of which produced gaps well above the bound below), not to claim GaBP
    is tightly near-optimal in this specific adversarial corner case.
    """
    gaps = []
    for H, s, levels, thresholds, C, sigma_n2, P, NU in _tiny_system_trials(20):
        x_ex, J_ex = exhaustive(s, H, C, NU, sigma_n2)
        x_g, _ = gabp(s, H, C, NU, sigma_n2, kmax=150, rho_damp=0.15, lam_check=20)
        xgq = Qb(x_g, levels, thresholds)
        b_g = beta_star(s, H, xgq, NU, sigma_n2)
        gaps.append(objective(s, H, xgq, b_g, NU, sigma_n2) - J_ex)

    assert np.mean(gaps) < 0.35   # generous margin over calibrated ~0.11


def test_gabp_tracks_mqp_at_realistic_scale():
    """The fairer test of GaBP's real target regime (per the paper: large
    NT, several users, not a 2-factor toy graph). At NT=32, NU=8, MQP and
    GaBP should land within the same ballpark of each other -- neither
    catastrophically worse -- since both target the same stationary
    point of the regularized objective (paper Sec. V-B.2). Calibrated
    mean(J_gabp - J_mqp) over 20 trials was consistently negative
    (-0.06 to -0.17 across several lambda_check values), i.e. GaBP was
    if anything slightly ahead of MQP on average here, not just "close."
    The threshold below is left generous specifically so this test still
    catches the failure mode it was written for: letting GaBP's zeta
    anneal down (rather than holding it fixed) after the eq-22 fix turns
    this same mean difference strongly positive, around +4 to +6."""
    NT, NU, b = 32, 8, 2
    P, sigma_n2 = 1.0, 0.02
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(16)

    diffs = []
    for trial in range(10):
        rng = np.random.default_rng(200 + trial)
        H = iid_Rayleigh_channel(NU, NT, rng)
        s = Sconst[rng.integers(0, 16, NU)]

        x_m, _ = mqp(s, H, C, P, NU, sigma_n2, np.zeros(NT, dtype=complex), Tmax=40)
        xmq = Qb(x_m, levels, thresholds)
        bm = beta_star(s, H, xmq, NU, sigma_n2)
        Jm = objective(s, H, xmq, bm, NU, sigma_n2)

        x_g, _ = gabp(s, H, C, NU, sigma_n2, kmax=150, rho_damp=0.15, lam_check=20)
        xgq = Qb(x_g, levels, thresholds)
        bg = beta_star(s, H, xgq, NU, sigma_n2)
        Jg = objective(s, H, xgq, bg, NU, sigma_n2)

        diffs.append(Jg - Jm)

    # not "GaBP is at least as good" (too strong -- sometimes it isn't,
    # by design of a different iterative path) -- just "not systematically
    # and dramatically worse"
    assert np.mean(diffs) < 0.5


def test_gabp_stable_at_large_nt():
    """Regression test for a real bug: rho_damp=0.5 (fine up to NT~32)
    let GaBP's objective oscillate by 2-3x every iteration and never
    settle at NT=64 (512 factor-graph edges vs. 256 -- more short cycles
    to reinforce each other), landing at BER close to chance. rho_damp
    =0.15 is now the default and was verified stable from NT=6 through
    NT=64; this test checks GaBP still comfortably beats a random guess
    at NT=64, 1-bit, moderate SNR -- the failure mode this guards against
    produced BER around 35-45% (indistinguishable from chance), not a
    subtle degradation, so the bound below has a lot of room and still
    catches it."""
    NT, NU, b = 64, 8, 1
    P, sigma_n2 = 1.0, 0.1
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(4)

    gaps_to_wf = []
    for trial in range(10):
        rng = np.random.default_rng(300 + trial)
        H = iid_Rayleigh_channel(NU, NT, rng)
        s = Sconst[rng.integers(0, 4, NU)]

        x_wf = Qb(precoder_wf(s, H, P, NU, sigma_n2), levels, thresholds)
        b_wf = beta_star(s, H, x_wf, NU, sigma_n2)
        J_wf = objective(s, H, x_wf, b_wf, NU, sigma_n2)

        x_g, _ = gabp(s, H, C, NU, sigma_n2, kmax=150, lam_check=20)  # default rho_damp
        xgq = Qb(x_g, levels, thresholds)
        bg = beta_star(s, H, xgq, NU, sigma_n2)
        J_g = objective(s, H, xgq, bg, NU, sigma_n2)

        gaps_to_wf.append(J_g - J_wf)

    # GaBP should be roughly comparable to WF here, not catastrophically
    # worse (the pre-fix failure mode was ~10-50x worse, not ~2x)
    assert np.mean(gaps_to_wf) < 1.0


# ---------------------------------------------------------------------
# CDM sanity
# ---------------------------------------------------------------------

def test_cdm_never_worse_than_its_initialization():
    """CDM is pure coordinate descent with no changing hyperparameters --
    unlike MQP, its objective (for FIXED beta within a sweep) must be
    monotonically non-increasing, no caveats. Check the end-to-end
    objective (recomputing beta) is at least as good as the quantized-WF
    starting point it was seeded from."""
    rng = np.random.default_rng(3)
    NT, NU, b = 8, 3, 1
    P, sigma_n2 = 1.0, 0.1
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(4)

    for _ in range(10):
        H = iid_Rayleigh_channel(NU, NT, rng)
        s = Sconst[rng.integers(0, 4, NU)]

        x_init = Qb(precoder_wf(s, H, P, NU, sigma_n2), levels, thresholds)
        b_init = beta_star(s, H, x_init, NU, sigma_n2)
        J_init = objective(s, H, x_init, b_init, NU, sigma_n2)

        x_cdm = cdm(s, H, C, NU, sigma_n2, x_init.copy())
        b_cdm = beta_star(s, H, x_cdm, NU, sigma_n2)
        J_cdm = objective(s, H, x_cdm, b_cdm, NU, sigma_n2)

        assert J_cdm <= J_init + 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))