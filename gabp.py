import numpy as np
from system_model import beta_star


def gabp(s, H, C, NU, sigma_n2, kmax=150, rho_damp=0.15, eps=1e-8,
         zeta0=None, zeta_min=None, lam_check=20.0, track_obj=False):
    """Algorithm 3. Returns (x_tilde, beta) -- the converged consensus
    estimate and its receiver scaling; apply Qb(...) yourself for the
    final hard output.

    zeta_min defaults to zeta0 -- i.e. NO graduated-non-convexity
    annealing by default, unlike mqp(). This is a deliberate, empirically
    forced choice, not an oversight: once gamma^2 = zeta/(diff2+zeta)^2
    is computed correctly , it grows toward
    1/zeta for an on-target antenna as zeta shrinks -- exactly as
    intended for MQP, where lambda_check is auto-tuned by the discrepancy
    principle to stay in balance with it. GaBP instead uses a fixed,
    "just large enough" lambda_check (per the paper's own guidance, Sec.
    IV, that no fine-tuning is needed). Combine a FIXED large lambda_check
    with a SHRINKING zeta and lambda_check*G_m can reach 1e6-1e7+ within
    the iteration budget, dwarfing the data precision by many orders of
    magnitude and making the denoiser snap to whatever point looks
    closest at that instant -- often before the data-driven consensus has
    actually settled, locking onto a wrong point it then can't escape
    (measured: mean objective gap to MQP went from about -0.1 to +5,
    i.e. from "matches" to "badly broken", purely from letting zeta
    anneal down after the gamma^2 fix). Holding zeta fixed and letting
    lambda_check alone control the discreteness pull keeps the intended
    "large enough to matter, insensitive beyond that" behavior the paper
    describes. Pass an explicit zeta_min < zeta0 only if you've verified
    it stays stable for your own (lambda_check, kmax, rho_damp) choice.

    rho_damp also needs to scale with the graph's size: rho_damp=0.5
    (fine at NT<=32) let the objective oscillate by 2-3x every iteration
    and never settle at NT=64 (512 edges instead of 256) -- the loopy
    graph simply has more short cycles to reinforce each other. rho_damp
    =0.15 was stable and matched MQP's ballpark from NT=6 (vs. exhaustive)
    through NT=64, and is now the default; if you push to even larger NT
    and see oscillation again (objective jumping by a large factor
    iteration to iteration when track_obj=True), lower it further before
    assuming anything else is wrong.
    """
    NU_, NT = H.shape
    if zeta0 is None:
        zeta0 = (np.abs(C.real).max() * 2) ** 2
    if zeta_min is None:
        zeta_min = zeta0  # see docstring above

    x_hat = np.zeros((NT, NU_), dtype=complex)  # edge messages, init per Alg.3
    v_hat = np.ones((NT, NU_))
    x_tilde = np.zeros(NT, dtype=complex)  # running consensus estimate
    obj_hist = [] if track_obj else None

    for t in range(kmax):
        zeta = zeta0 * (zeta_min / zeta0) ** min(1, t / max(kmax - 1, 1))
        x_hat_prev = x_hat  # for eps_k, per Alg.3

        # Step 1: soft interference cancellation, eqs (35)-(36).
        # full_pred[n] = sum_m H[n,m]*x_hat[m,n]; full_var[n] likewise --
        # computed directly via einsum rather than np.diag(H @ x_hat),
        # which would needlessly form the whole (NU,NU) product just to
        # keep its diagonal (O(NU^2*NT) instead of O(NU*NT), the
        # complexity Table I actually claims for this step).
        full_pred = np.einsum('nm,mn->n', H, x_hat)  # (NU,)
        s_tilde = (s - full_pred)[None, :] + H.T * x_hat  # (NT,NU)
        full_var = np.einsum('nm,mn->n', np.abs(H) ** 2, v_hat)  # (NU,)
        psi = full_var[None, :] - (np.abs(H) ** 2).T * v_hat + sigma_n2  # (NT,NU)

        # discreteness prior, using the PREVIOUS iteration's consensus x_tilde
        diff2 = np.abs(x_tilde[None, :] - C[:, None]) ** 2  # (2^2b,NT)
        gamma2 = zeta / (diff2 + zeta) ** 2  # eq (22), fixed
        G_m = gamma2.sum(axis=0)  # (NT,)
        g_m = (gamma2 * C[:, None]).sum(axis=0)  # (NT,)

        # Step 2: aggregate belief, eqs (37)-(38) -- TRUE leave-one-out
        w_edge = (np.abs(H) ** 2).T / psi  # per-edge |h|^2/psi
        num_edge = np.conj(H).T * s_tilde / psi  # per-edge h^H*s~/psi
        w_full = w_edge.sum(axis=1, keepdims=True)  # sum over ALL factors
        num_full = num_edge.sum(axis=1, keepdims=True)
        w_loo = w_full - w_edge  # exclude factor n
        num_loo = num_full - num_edge
        v_bar = 1.0 / w_loo
        y_bar = v_bar * num_loo

        # denoiser: fuse data belief + implicit unit prior + discreteness
        # prior, eqs (39)-(42)
        denom = 1.0 / v_bar + 1.0 + lam_check * G_m[:, None]
        numer = y_bar / v_bar + lam_check * g_m[:, None]
        x_new = numer / denom
        v_new = 1.0 / denom

        # damping, eqs (45)-(46)
        x_hat = rho_damp * x_new + (1 - rho_damp) * x_hat
        v_hat = rho_damp * v_new + (1 - rho_damp) * v_hat

        # belief consensus, eqs (47)-(48) -- full sum, no exclusion
        v_tilde = 1.0 / w_full[:, 0]
        x_tilde = v_tilde * num_full[:, 0]

        if track_obj:
            from system_model import objective
            b = beta_star(s, H, x_tilde, NU, sigma_n2)
            obj_hist.append(objective(s, H, x_tilde, b, NU, sigma_n2))

        # convergence check per Alg. 3: eps_k = ||x_hat^(k+1) - x_hat^(k)||_2
        # (the edge messages, not the consensus estimate x_tilde -- they
        # track together in practice, but this matches what's specified)
        if np.linalg.norm(x_hat - x_hat_prev) < eps:
            break

    beta = beta_star(s, H, x_tilde, NU, sigma_n2)
    if track_obj:
        return x_tilde, beta, obj_hist
    return x_tilde, beta

