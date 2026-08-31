"""
mqp.py -- Multibit Quantized Precoding, Algorithm 2 (eqs 18-32).

Implementation note: lambda_check (the renormalized lambda-hat of eq 28)
is tracked directly throughout -- the paper's working equations (28),
(30), (31) are all stated in terms of lambda_check, so there is no need
to separately track the un-checked lambda of eq (19).

GOTCHA: initializing x = 0 makes beta_star's numerator AND denominator
both come from Hx=0, giving beta = 0/den = 0 -- and the very next line,
s_check = s/beta, then divides by zero. Seed beta=1.0 for a zero x_init;
the real beta* formula (eq 9) takes over from iteration 1 onward.

KNOWN OPEN LIMITATION, found while building the experiments/ scripts:
the default zeta0/zeta_min/kappa/lambda_min/lambda_max heuristics below
work well for 1-bit alphabets (verified against exhaustive search, and
against WF at NT up to 64) but can settle into a local optimum clearly
WORSE than plain quantized WF for 2-bit+ alphabets combined with
higher-order modulation (16-QAM), at both small (NT=6) and large (NT=64)
antenna counts. Confirmed NOT fixed by: more iterations, WF instead of
zero initialization, a smaller kappa, or scaling zeta0 by the alphabet's
point-spacing instead of its span. This looks like it needs a genuinely
different scaling law for the (zeta, lambda) schedule at higher
resolutions, not a one-line tweak -- treat any 2-bit+ result from this
implementation with real skepticism until you've either fixed this or
verified it against exhaustive search / a baseline for your specific
config. 1-bit configurations were extensively cross-validated and are
solid (see tests/test_vs_exhaustive.py).
"""

import numpy as np
from system_model import beta_star


def mqp(s, H, C, P, NU, sigma_n2, x_init, Tmax=30,
        zeta0=None, zeta_min=None, lam_min=1e-6, lam_max=1e6, kappa=0.5,
        track_obj=False):
    """Algorithm 2. Returns (x, beta) -- the UNQUANTIZED iterate and its
    receiver scaling; apply Qb(x, levels, thresholds) yourself for the
    final hard output (Algorithm 2, line 9)."""
    NT = H.shape[1]
    x = x_init.copy()
    beta = 1.0 if np.allclose(x, 0) else beta_star(s, H, x, NU, sigma_n2)
    delta_star = NU * sigma_n2 / P                        # noise floor, eq (31)

    if zeta0 is None:
        zeta0 = (np.abs(C.real).max() * 2) ** 2            # ~ squared alphabet span
    if zeta_min is None:
        zeta_min = zeta0 * 1e-5

    lam_check = 1.0
    obj_hist = [] if track_obj else None

    for t in range(Tmax):
        # graduated non-convexity schedule, eq (32)
        zeta = zeta0 * (zeta_min / zeta0) ** min(1, t / max(Tmax - 1, 1))

        # FP auxiliary weights, eq (22): gamma_{i,j} = sqrt(zeta)/(diff2+zeta),
        # so gamma_{i,j}^2 = zeta/(diff2+zeta)^2 -- NOT zeta/(diff2+zeta)^2 * zeta.
        # (That extra trailing "* zeta" was a real bug in an earlier version of
        # this file: due to operator precedence, `zeta / (diff2+zeta)**2 * zeta`
        # evaluates left-to-right as `(zeta/(diff2+zeta)**2) * zeta`, giving
        # zeta^2/(diff2+zeta)^2 instead. At diff2=0 that pins gamma^2 at exactly
        # 1 for every zeta, instead of 1/zeta -- which silently disables the
        # "sharpen as zeta shrinks" mechanism that graduated non-convexity
        # depends on. Caught by re-deriving eq 22 by hand against the code.)
        diff2 = np.abs(x[None, :] - C[:, None]) ** 2        # (2^2b, NT)
        gamma2 = zeta / (diff2 + zeta) ** 2

        # pack into g, G, eqs (25)-(26)
        G_diag = gamma2.sum(axis=0)
        g = (gamma2 * C[:, None]).sum(axis=0)

        # closed-form least-squares update, eq (30) -- via the Woodbury
        # identity rather than a direct NT x NT solve, matching Table I's
        # claimed O(N_U^2*N_T) per iteration (a direct solve of the full
        # system matrix is O(N_T^3) and was the actual cost of an earlier
        # version of this function -- functionally correct, but silently
        # not what the paper's complexity analysis describes).
        #
        # A = lam_check*G + delta_star*I  (diagonal, so A^{-1} is trivial)
        # (A + H^H H)^{-1} = A^{-1} - A^{-1}H^H (I_NU + H A^{-1} H^H)^{-1} H A^{-1}
        # -- the only real inversion left is of the NU x NU matrix M below.
        s_check = s / beta
        a_inv = 1.0 / (lam_check * G_diag + delta_star)          # (NT,)
        r = H.conj().T @ s_check + lam_check * g                  # (NT,)
        Ainv_r = a_inv * r
        H_scaled = H * a_inv[None, :]                              # H @ diag(a_inv)
        M = np.eye(NU) + H_scaled @ H.conj().T                     # (NU, NU)
        y = np.linalg.solve(M, H @ Ainv_r)
        x = Ainv_r - a_inv * (H.conj().T @ y)

        # Morozov discrepancy update, eq (31)
        F = np.linalg.norm(s_check - H @ x) ** 2
        lam_check = np.clip(lam_check * (delta_star / max(F, 1e-12)) ** kappa,
                             lam_min, lam_max)

        # refresh receiver scaling, eq (9)
        beta = beta_star(s, H, x, NU, sigma_n2)

        if track_obj:
            from system_model import objective
            obj_hist.append(objective(s, H, x, beta, NU, sigma_n2))

    if track_obj:
        return x, beta, obj_hist
    return x, beta