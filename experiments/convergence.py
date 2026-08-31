"""
experiments/fig8_convergence.py -- reproduces Fig. 8: objective value per
iteration for MQP, GaBP, and CDM on a fixed problem instance.

Uses NT=64, NU=8, QPSK, 1-bit (see mqp.py's documented 2-bit limitation).

NOTE on what you'll see for MQP: tutorial Sec. 3.7 explains why MQP's
raw objective trace is NOT expected to be monotone while zeta and
lambda_check are still adapting -- only once they've settled does the
majorization-minimization monotonicity guarantee apply. Don't mistake an
early bump for a bug; check whether it's still bumping in the LAST few
iterations (that WOULD be worth investigating).

Run:  python3 experiments/fig8_convergence.py
Output: fig8_convergence.png in this directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from system_model import gen_alphabet, qam_constellation, beta_star, objective, Qb, iid_Rayleigh_channel
from baselines import precoder_wf , cdm 
from mqp import mqp
from gabp import gabp

NT, NU, M, b, P = 64, 8, 4, 1, 1.0
SNR_DB = 10
TMAX = 35


def cdm_trace(s, H, C, NU, sigma_n2, x_init, Tmax):
    """CDM re-implemented locally to record the objective after every
    full sweep (cdm.py's own function only returns the final x)."""
    NT = H.shape[1]
    x = x_init.copy()
    z = H @ x
    beta = beta_star(s, H, x, NU, sigma_n2)
    hist = []
    for _ in range(Tmax):
        for n in range(NT):
            z_try = z[None, :] + (C[:, None] - x[n]) * H[:, n][None, :]
            Jvals = (np.sum(np.abs(s[None, :] - beta * z_try) ** 2, axis=1)
                     + beta ** 2 * NU * sigma_n2)
            c_star = C[np.argmin(Jvals)]
            z = z + (c_star - x[n]) * H[:, n]
            x[n] = c_star
        beta = beta_star(s, H, x, NU, sigma_n2)
        hist.append(objective(s, H, x, beta, NU, sigma_n2))
    return hist


if __name__ == "__main__":
    sigma_n2 = P / (10 ** (SNR_DB / 10))
    levels, thresholds, C = gen_alphabet(b, P, NT)
    Sconst = qam_constellation(M)
    rng = np.random.default_rng(3)
    H = iid_Rayleigh_channel(NU, NT, rng)
    s = Sconst[rng.integers(0, M, NU)]

    print(f"NT={NT}, NU={NU}, QPSK, 1-bit, SNR={SNR_DB}dB, Tmax={TMAX}")

    _, _, mqp_hist = mqp(s, H, C, P, NU, sigma_n2, np.zeros(NT, dtype=complex),
                          Tmax=TMAX, track_obj=True)
    _, _, gabp_hist = gabp(s, H, C, NU, sigma_n2, kmax=TMAX, eps=0.0, track_obj=True)

    x_wf = Qb(precoder_wf(s, H, P, NU, sigma_n2), levels, thresholds)
    cdm_hist = cdm_trace(s, H, C, NU, sigma_n2, x_wf.copy(), TMAX)

    print("MQP  last 5:", np.round(mqp_hist[-5:], 4))
    print("GaBP last 5:", np.round(gabp_hist[-5:], 4))
    print("CDM  last 5:", np.round(cdm_hist[-5:], 4))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.semilogy(range(1, TMAX + 1), cdm_hist, "o-", color="tab:blue", label="CDM")
    ax.semilogy(range(1, TMAX + 1), gabp_hist, "o--", color="tab:red", label="Prop. GaBP")
    ax.semilogy(range(1, TMAX + 1), mqp_hist, "o-", color="tab:red", label="Prop. MQP")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Convergence, NT={NT}, NU={NU}, QPSK, 1-bit, SNR={SNR_DB}dB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig8_convergence.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")