import numpy as np
from system_model import *

# 5. CLASSICAL LINEAR BASELINES  (Sec. III-A.1-3, eqs 11-17)

# Implementation note: rather than reproducing the trace-formula power
# scalars (12)/(14)/(16) verbatim, we compute the *unnormalized* precoder
# and rescale so ||x||^2 = P exactly. These are mathematically identical
# (there is only one rho that hits the power budget) and far less
# error-prone to code.

def precoder_wf(s, H, P, NU, sigma_n2):
    """Wiener filter / MMSE precoder, eq (11)."""
    NT = H.shape[1]
    A = H.conj().T @ H + (NU * sigma_n2 / P) * np.eye(NT)
    x_raw = np.linalg.solve(A, H.conj().T @ s)
    return x_raw / (np.linalg.norm(x_raw) / np.sqrt(P))


def precoder_mrt(s, H, P):
    """Maximum ratio transmission (matched filter), eq (13)."""
    x_raw = H.conj().T @ s
    return x_raw / (np.linalg.norm(x_raw) / np.sqrt(P))


def precoder_zf(s, H, P):
    """Zero-forcing precoder, eq (15)."""
    HHH = H @ H.conj().T
    x_raw = H.conj().T @ np.linalg.solve(HHH, s)
    return x_raw / (np.linalg.norm(x_raw) / np.sqrt(P))

# 6. COORDINATE DESCENT PRECODING  (Algorithm 1)

def cdm(s, H, C, NU, sigma_n2, x_init, Tmax=15):
    """Algorithm 1. x_init should typically be a quantized WF/MMSE
    solution (CDM needs a decent starting point to perform well)."""
    NT = H.shape[1]
    x = x_init.copy()
    z = H @ x                       # noiseless received vector, tracked incrementally
    beta = beta_star(s, H, x, NU, sigma_n2)
    for _ in range(Tmax):
        x_old = x.copy()
        for n in range(NT):
            z_try = z[None, :] + (C[:, None] - x[n]) * H[:, n][None, :]
            Jvals = (np.sum(np.abs(s[None, :] - beta * z_try) ** 2, axis=1)
                     + beta ** 2 * NU * sigma_n2)
            c_star = C[np.argmin(Jvals)]
            z = z + (c_star - x[n]) * H[:, n]              # rank-one update
            x[n] = c_star
        beta = beta_star(s, H, x, NU, sigma_n2)
        if np.allclose(x, x_old):
            break
    return x