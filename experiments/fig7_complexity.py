"""
experiments/fig7_complexity.py -- reproduces Fig. 7's story: empirical
wall-clock cost per precoding call vs. NT, for the proposed and baseline
methods. Table I gives the paper's asymptotic complexity orders; this
script measures actual runtime rather than counting operations, since
wall-clock is simpler to get right and the qualitative scaling story
(MQP and GaBP roughly linear in NT vs. WF's direct-solve growing faster,
GaBP eventually pulling ahead of MQP as NT grows) is what matters here,
not matching an exact multiplier.

Run:  python3 experiments/fig7_complexity.py
Output: fig7_complexity.png in this directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt

from system_model import gen_alphabet, qam_constellation, iid_Rayleigh_channel
from baselines import precoder_wf , cdm
from mqp import mqp
from gabp import gabp

NT_RANGE = [8, 16, 32, 64, 128, 256]
NU = 8
M, b, P = 4, 1, 1.0   # QPSK, 1-bit -- see mqp.py's documented 2-bit limitation
N_REPEATS = 15


def time_call(fn, n_repeats):
    t0 = time.time()
    for _ in range(n_repeats):
        fn()
    return (time.time() - t0) / n_repeats


def run():
    Sconst = qam_constellation(M)
    times = {"WF": [], "CDM": [], "MQP": [], "GaBP": []}

    for NT in NT_RANGE:
        levels, thresholds, C = gen_alphabet(b, P, NT)
        rng = np.random.default_rng(0)
        H = iid_Rayleigh_channel(NU, NT, rng)
        s = Sconst[rng.integers(0, M, NU)]
        sigma_n2 = 0.05

        times["WF"].append(time_call(lambda: precoder_wf(s, H, P, NU, sigma_n2), N_REPEATS))
        x_wf = precoder_wf(s, H, P, NU, sigma_n2)
        times["CDM"].append(time_call(lambda: cdm(s, H, C, NU, sigma_n2, x_wf.copy()), N_REPEATS))
        times["MQP"].append(time_call(
            lambda: mqp(s, H, C, P, NU, sigma_n2, np.zeros(NT, dtype=complex), Tmax=30), N_REPEATS))
        times["GaBP"].append(time_call(
            lambda: gabp(s, H, C, NU, sigma_n2, kmax=150), max(3, N_REPEATS // 3)))

        print(f"NT={NT:4d}:  WF={times['WF'][-1]*1e3:8.3f}ms  CDM={times['CDM'][-1]*1e3:8.3f}ms  "
              f"MQP={times['MQP'][-1]*1e3:8.3f}ms  GaBP={times['GaBP'][-1]*1e3:8.3f}ms")

    return times


if __name__ == "__main__":
    print(f"NU={NU} fixed, QPSK, 1-bit, sweeping NT -- {N_REPEATS} repeats per point")
    times = run()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    styles = {"WF": "tab:blue", "CDM": "tab:orange", "MQP": "tab:red", "GaBP": "tab:green"}
    for m, color in styles.items():
        ax.loglog(NT_RANGE, np.array(times[m]) * 1e3, "o-", color=color, label=m)
    ax.set_xlabel("Number of transmit antennas NT")
    ax.set_ylabel("Wall-clock time per call [ms]")
    ax.set_title(f"Runtime scaling vs. NT (NU={NU}, QPSK, 1-bit)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig7_complexity.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
