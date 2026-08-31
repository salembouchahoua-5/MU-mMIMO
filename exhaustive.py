import itertools
import numpy as np

def exhaustive(s , H , C ,NU , sigma_n2) :
    NT = H.shape[1]
    X = np.array(list(itertools.product(C , repeat = NT)))

    Hx_all = X @ H.T

    beta_num = np.real(np.sum((np.conj(s)[None , :] * Hx_all) , axis = 1))
    beta_den = np.sum(np.abs(Hx_all) ** 2, axis=1) + NU * sigma_n2    # Avoids np.linalg.norm's redundant square-root calculation and function overhead for maximum vectorization speed.
    beta_all = beta_num / beta_den

    resid = s[None , :] - beta_all[: , None] * Hx_all
    J_all = np.sum(np.abs(resid) ** 2 , axis = 1) + beta_all**2 * NU * sigma_n2

    best_idx = np.argmin(J_all)

    return X[best_idx] , J_all[best_idx]