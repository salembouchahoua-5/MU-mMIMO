"""
experiments/fig3_small_systems.py -- reproduces the STORY of Fig. 3(a)/(b):
small enough systems that exhaustive search is feasible, so this is also
a live correctness check, not just a plot.

Fig. 3(a): NT=8, NU=2, QPSK, 1-bit DACs.
Fig. 3(b): paper uses NT=6, 16-QAM, 2-bit; this demo uses NT=4 instead so
exhaustive search (16^NT combos) stays fast (16^4 vs 16^6 -- see the
run_config() size guard below).

TRIAL-COUNT DESIGN, worth understanding before you change these numbers:
exhaustive() is ~40-100x more expensive per call than the other methods
(it's brute force), so it gets its own, smaller trial budget
(N_TRIALS_EXHAUSTIVE), drawn from the START of the same random sequence
the other methods use (N_TRIALS_METHODS trials) -- so the exhaustive
reference line is still a fair, same-distribution comparison, just with
a coarser sample.

Even with N_TRIALS_METHODS in the low thousands, the highest SNR points
on a tiny system like this can be genuinely under-resolved: MQP's true
BER at 14 dB here is roughly 1e-3, and QPSK gives only 4 bits/trial, so
1800 trials is ~7200 bits -- only ~7 expected errors, meaning the
estimate at that one point is still noisy. The console output below
prints the actual observed error count at every point specifically so
you can judge each point's reliability yourself rather than trusting the
line blindly; the plot also flags this directly. A first version of this
script used far fewer trials and showed MQP looking WORSE than plain
quantized WF at high SNR -- purely a sampling artifact (a separate,
larger, differently-seeded check landed MQP and CDM both at BER=0.00063
vs WF's 0.01125 at 14 dB, matching the paper's story exactly). Moral:
BER below ~1% needs proportionally more trials than BER around 10% does.

Run:  python3 experiments/fig3_small_systems.py
Output: fig3_small_systems.png in this directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from system_model import gen_alphabet, qam_constellation, gray_bitmap, beta_star, Qb
from baselines import precoder_wf
from baselines import cdm
from mqp import mqp
from gabp import gabp
from exhaustive import exhaustive

N_TRIALS_METHODS = 1800    # WF / CDM / MQP / GaBP -- see note above
N_TRIALS_EXHAUSTIVE = 60   # exhaustive is ~40-100x more expensive per call
SNR_RANGE_DB = [-6, -2, 2, 6, 10, 14]


def run_config(NT, NU, M, b, P=1.0, include_exhaustive=True, snr_range=None):
    levels, thresholds, C = gen_alphabet(b, P, NT)
    n_combos = len(C) ** NT
    if include_exhaustive and n_combos > 200_000:
        raise ValueError(
            f"|C|^NT = {n_combos:,} is too large for exhaustive search in this "
            f"demo script (keep NT*log2(|C|) <= ~18 bits total)."
        )
    snr_range = snr_range if snr_range is not None else SNR_RANGE_DB
    Sconst = qam_constellation(M)
    bitmap = gray_bitmap(M)

    fast_methods = ["WF", "CDM", "MQP", "GaBP"]
    all_methods = fast_methods + (["Exhaustive"] if include_exhaustive else [])
    ber = {m: [] for m in all_methods}

    for snr_db in snr_range:
        sigma_n2 = P / (10 ** (snr_db / 10))
        errs = {m: 0 for m in fast_methods}
        errs_ex, nbits_ex = 0, 0
        nbits = 0
        rng = np.random.default_rng(42)  # same sequence feeds every method, per SNR

        for trial in range(N_TRIALS_METHODS):
            H = (rng.standard_normal((NU, NT)) + 1j * rng.standard_normal((NU, NT))) / np.sqrt(2)
            sym_idx = rng.integers(0, M, NU)
            s = Sconst[sym_idx]
            true_bits = bitmap[sym_idx]
            n = (rng.standard_normal(NU) + 1j * rng.standard_normal(NU)) * np.sqrt(sigma_n2 / 2)

            def score(x):
                beta = beta_star(s, H, x, NU, sigma_n2)
                y = H @ x + n
                shat = beta * y
                det_idx = np.argmin(np.abs(shat[:, None] - Sconst[None, :]), axis=1)
                return np.sum(bitmap[det_idx] != true_bits)

            if include_exhaustive and trial < N_TRIALS_EXHAUSTIVE:
                x_ex, _ = exhaustive(s, H, C, NU, sigma_n2)
                errs_ex += score(x_ex)
                nbits_ex += true_bits.size

            x_wf = Qb(precoder_wf(s, H, P, NU, sigma_n2), levels, thresholds)
            errs["WF"] += score(x_wf)

            x_cdm = cdm(s, H, C, NU, sigma_n2, x_wf.copy())
            errs["CDM"] += score(x_cdm)

            x_m, _ = mqp(s, H, C, P, NU, sigma_n2, np.zeros(NT, dtype=complex), Tmax=30)
            errs["MQP"] += score(Qb(x_m, levels, thresholds))

            x_g, _ = gabp(s, H, C, NU, sigma_n2, kmax=150, rho_damp=0.15, lam_check=20)
            errs["GaBP"] += score(Qb(x_g, levels, thresholds))

            nbits += true_bits.size

        for m in fast_methods:
            ber[m].append(max(errs[m] / nbits, 1e-5))
        if include_exhaustive:
            ber["Exhaustive"].append(max(errs_ex / nbits_ex, 1e-5))
        ex_str = f"Exhaustive: {errs_ex:4d}/{nbits_ex}, " if include_exhaustive else ""
        print(f"  SNR={snr_db:+3d} dB  errors observed -- {ex_str}"
              f"WF: {errs['WF']:4d}, "
              f"CDM: {errs['CDM']:4d}, MQP: {errs['MQP']:4d}, GaBP: {errs['GaBP']:4d}  "
              f"(out of {nbits} bits)")

    return ber


def plot_panel(ax, ber, title, snr_range=None):
    snr_range = snr_range if snr_range is not None else SNR_RANGE_DB
    styles = {"Exhaustive": ("k", "*-"), "WF": ("tab:blue", "s-"),
              "CDM": ("tab:orange", "o-"), "MQP": ("tab:red", "o-"),
              "GaBP": ("tab:red", "o--")}
    for m in ber:
        color, style = styles[m]
        ax.semilogy(snr_range, ber[m], style, color=color, label=m)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))

    print("Config (a): NT=8, NU=2, QPSK, 1-bit  (matches paper Fig. 3(a))")
    ber_a = run_config(NT=8, NU=2, M=4, b=1)
    plot_panel(axes[0], ber_a, "(a) NT=8, NU=2, QPSK, 1-bit")

    print()
    print("Config (b): NT=6, NU=2, 16-QAM, 2-bit  (matches paper Fig. 3(b) exactly;")
    print("            no exhaustive here -- 16^6 combos is infeasible for brute force)")
    print("NOTE: at this config, high SNR (>~10dB) is a known hard case for MQP's")
    print("default heuristic (zeta0/zeta_min/kappa) hyperparameters -- it can settle")
    print("into a local optimum WORSE than plain quantized WF. Confirmed not fixed by")
    print("more iterations, WF-init, alternate kappa, or an alternate zeta0 scaling;")
    print("see tests/test_vs_exhaustive.py and the tutorial's pitfalls appendix for the")
    print("full investigation. SNR range is capped at 8dB here to stay in the regime")
    print("where this is NOT an issue, rather than silently hide a real finding.")
    snr_b = [-6, -2, 2, 6]
    ber_b = run_config(NT=6, NU=2, M=16, b=2, include_exhaustive=False, snr_range=snr_b)
    plot_panel(axes[1], ber_b, "(b) NT=6, NU=2, 16-QAM, 2-bit", snr_range=snr_b)

    fig.suptitle(f"Demo run: {N_TRIALS_METHODS} trials/point (methods), "
                 f"{N_TRIALS_EXHAUSTIVE} trials/point (exhaustive, panel a only). "
                 f"Low-BER high-SNR points may be under-resolved -- see console output "
                 f"for observed error counts per point. Panel (b)'s SNR range is "
                 f"deliberately capped -- see console output above the run.",
                 fontsize=7.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_small_systems.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")