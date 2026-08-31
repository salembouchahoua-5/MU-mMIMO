"""
experiments/fig6_csi_error.py -- reproduces Fig. 6(a): robustness to
channel estimation error, eq (49): Hhat = sqrt(1-tau^2)*H + tau*E.

Uses NT=64, NU=8, QPSK, 1-bit -- the config MQP/GaBP are solidly
verified for (see mqp.py's docstring on the open 2-bit limitation).

Run:  python3 experiments/fig6_csi_error.py
Output: fig6_csi_error.png in this directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from system_model import (gen_alphabet, qam_constellation, gray_bitmap,
                           beta_star, Qb, iid_Rayleigh_channel, apply_csi_error)
from baselines import precoder_wf , cdm 
from mqp import mqp
from gabp import gabp

N_TRIALS = 250
TAU_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
NT, NU, M, b, P = 64, 8, 4, 1, 1.0
SNR_DB = 10  # fixed operating point, matching the paper's use of a single SNR here


def run():
    sigma_n2 = P / (10 ** (SNR_DB / 10))
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(M)
    bitmap = gray_bitmap(M)
    methods = ["WF", "CDM", "MQP", "GaBP"]
    ber = {m: [] for m in methods}

    for tau in TAU_RANGE:
        errs = {m: 0 for m in methods}
        nbits = 0
        rng = np.random.default_rng(11)

        for _ in range(N_TRIALS):
            H_true = iid_Rayleigh_channel(NU, NT, rng)
            H_est = apply_csi_error(H_true, tau, rng)  # precoder sees this
            sym_idx = rng.integers(0, M, NU)
            s = Sconst[sym_idx]
            true_bits = bitmap[sym_idx]
            n = (rng.standard_normal(NU) + 1j * rng.standard_normal(NU)) * np.sqrt(sigma_n2 / 2)

            def score(x):
                # precoder designed against H_est, but the signal actually
                # propagates through the TRUE channel H_true
                beta = beta_star(s, H_est, x, NU, sigma_n2)
                y = H_true @ x + n
                shat = beta * y
                det_idx = np.argmin(np.abs(shat[:, None] - Sconst[None, :]), axis=1)
                return np.sum(bitmap[det_idx] != true_bits)

            x_wf = Qb(precoder_wf(s, H_est, P, NU, sigma_n2), levels, thresholds)
            errs["WF"] += score(x_wf)

            x_cdm = cdm(s, H_est, C, NU, sigma_n2, x_wf.copy())
            errs["CDM"] += score(x_cdm)

            x_m, _ = mqp(s, H_est, C, P, NU, sigma_n2, np.zeros(NT, dtype=complex), Tmax=30)
            errs["MQP"] += score(Qb(x_m, levels, thresholds))

            x_g, _ = gabp(s, H_est, C, NU, sigma_n2, kmax=150, rho_damp=0.15, lam_check=20)
            errs["GaBP"] += score(Qb(x_g, levels, thresholds))

            nbits += true_bits.size

        for m in methods:
            ber[m].append(max(errs[m] / nbits, 1e-5))
        print(f"  tau={tau:.2f}  errors -- WF:{errs['WF']:5d} CDM:{errs['CDM']:5d} "
              f"MQP:{errs['MQP']:5d} GaBP:{errs['GaBP']:5d}  (of {nbits} bits)")

    return ber


if __name__ == "__main__":
    print(f"NT={NT}, NU={NU}, QPSK, 1-bit, SNR={SNR_DB}dB, sweeping CSI error tau")
    ber = run()

    fig, ax = plt.subplots(figsize=(6, 5))
    styles = {"WF": ("tab:blue", "s-"), "CDM": ("tab:orange", "o-"),
              "MQP": ("tab:red", "o-"), "GaBP": ("tab:red", "o--")}
    for m, (color, style) in styles.items():
        ax.semilogy(TAU_RANGE, ber[m], style, color=color, label=m)
    ax.set_xlabel("CSI Relative Error (tau)")
    ax.set_ylabel("Average BER")
    ax.set_title(f"Channel estimation impairment, NT={NT}, NU={NU}, SNR={SNR_DB}dB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig6_csi_error.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")