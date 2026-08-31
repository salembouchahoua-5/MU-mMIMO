"""
experiments/fig4_large_systems.py -- reproduces Fig. 4(a)/(b): NT=64,
NU=8, no exhaustive search shown (infeasible at this scale, exactly as
in the paper -- complexity grows as 2^(b*NT)).

Fig. 4(a): QPSK, 1-bit.
Fig. 4(b): 16-QAM, 2-bit.

Run:  python3 experiments/fig4_large_systems.py
Output: fig4_large_systems.png in this directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from system_model import gen_alphabet, qam_constellation, gray_bitmap, beta_star, Qb
from baselines import precoder_wf , cdm
from mqp import mqp
from gabp import gabp

N_TRIALS = 300
SNR_RANGE_DB = [-6, -2, 2, 6, 10, 14]


def run_config(NT, NU, M, b, P=1.0, snr_range=None):
    snr_range = snr_range if snr_range is not None else SNR_RANGE_DB
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(M)
    bitmap = gray_bitmap(M)
    methods = ["WF", "CDM", "MQP", "GaBP"]
    ber = {m: [] for m in methods}

    for snr_db in snr_range:
        sigma_n2 = P / (10 ** (snr_db / 10))
        errs = {m: 0 for m in methods}
        nbits = 0
        rng = np.random.default_rng(7)

        for _ in range(N_TRIALS):
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

            x_wf = Qb(precoder_wf(s, H, P, NU, sigma_n2), levels, thresholds)
            errs["WF"] += score(x_wf)

            x_cdm = cdm(s, H, C, NU, sigma_n2, x_wf.copy())
            errs["CDM"] += score(x_cdm)

            x_m, _ = mqp(s, H, C, P, NU, sigma_n2, np.zeros(NT, dtype=complex), Tmax=30)
            errs["MQP"] += score(Qb(x_m, levels, thresholds))

            x_g, _ = gabp(s, H, C, NU, sigma_n2, kmax=150, rho_damp=0.15, lam_check=20)
            errs["GaBP"] += score(Qb(x_g, levels, thresholds))

            nbits += true_bits.size

        for m in methods:
            ber[m].append(max(errs[m] / nbits, 1e-5))
        print(f"  SNR={snr_db:+3d} dB  errors -- WF:{errs['WF']:5d} CDM:{errs['CDM']:5d} "
              f"MQP:{errs['MQP']:5d} GaBP:{errs['GaBP']:5d}  (of {nbits} bits)")

    return ber


def plot_panel(ax, ber, title, snr_range=None):
    snr_range = snr_range if snr_range is not None else SNR_RANGE_DB
    styles = {"WF": ("tab:blue", "s-"), "CDM": ("tab:orange", "o-"),
              "MQP": ("tab:red", "o-"), "GaBP": ("tab:red", "o--")}
    for m, (color, style) in styles.items():
        ax.semilogy(snr_range, ber[m], style, color=color, label=m)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    print("Config (a): NT=64, NU=8, QPSK, 1-bit  (matches paper Fig. 4(a))")
    ber_a = run_config(NT=64, NU=8, M=4, b=1)
    plot_panel(axes[0], ber_a, "(a) NT=64, NU=8, QPSK, 1-bit")

    print("Config (b): NT=64, NU=8, 16-QAM, 2-bit  (matches paper Fig. 4(b))")
    ber_b = run_config(NT=64, NU=8, M=16, b=2)
    plot_panel(axes[1], ber_b, "(b) NT=64, NU=8, 16-QAM, 2-bit")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig4_large_systems.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")