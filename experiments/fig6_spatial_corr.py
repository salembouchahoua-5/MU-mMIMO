"""
experiments/fig6_spatial_corr.py -- reproduces Fig. 6(b): robustness to
spatially correlated channels via the Kronecker/Jakes model, eqs (50)-(51).

Uses NT=16, NU=4, QPSK, 1-bit at a fixed SNR (matching the paper's use of
a fixed operating SNR for this sweep, though the paper itself uses 16-QAM
2-bit here -- switched to 1-bit given mqp.py's documented open limitation
at higher resolutions; the qualitative story -- rising BER as spacing
shrinks and correlation grows -- doesn't depend on that choice).

Run:  python3 experiments/fig6_spatial_corr.py
Output: fig6_spatial_corr.png in this directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from system_model import (gen_alphabet, qam_constellation, gray_bitmap,
                           beta_star, Qb, kronecker_correlated_channel)
from baselines import precoder_wf , cdm
from mqp import mqp
from gabp import gabp

N_TRIALS = 250
SPACING_RANGE = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
NT, NU, M, b, P = 16, 4, 4, 1, 1.0
SNR_DB = 15  # matches the paper's fixed operating point for this sweep


def run():
    sigma_n2 = P / (10 ** (SNR_DB / 10))
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(M)
    bitmap = gray_bitmap(M)
    methods = ["WF", "CDM", "MQP", "GaBP"]
    ber = {m: [] for m in methods}

    for d_lambda in SPACING_RANGE:
        errs = {m: 0 for m in methods}
        nbits = 0
        rng = np.random.default_rng(23)

        for _ in range(N_TRIALS):
            H = kronecker_correlated_channel(NU, NT, d_lambda, rng)
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
        print(f"  d/lambda={d_lambda:.2f}  errors -- WF:{errs['WF']:5d} CDM:{errs['CDM']:5d} "
              f"MQP:{errs['MQP']:5d} GaBP:{errs['GaBP']:5d}  (of {nbits} bits)")

    return ber


if __name__ == "__main__":
    print(f"NT={NT}, NU={NU}, QPSK, 1-bit, SNR={SNR_DB}dB, sweeping antenna spacing d/lambda")
    ber = run()

    fig, ax = plt.subplots(figsize=(6, 5))
    styles = {"WF": ("tab:blue", "s-"), "CDM": ("tab:orange", "o-"),
              "MQP": ("tab:red", "o-"), "GaBP": ("tab:red", "o--")}
    for m, (color, style) in styles.items():
        ax.semilogy(SPACING_RANGE, ber[m], style, color=color, label=m)
    ax.set_xlabel("Antenna Spacing d/lambda")
    ax.set_ylabel("Average BER")
    ax.set_title(f"Spatial correlation impairment, NT={NT}, NU={NU}, SNR={SNR_DB}dB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig6_spatial_corr.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")