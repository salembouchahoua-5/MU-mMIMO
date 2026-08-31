import numpy as np
from system_model import gen_alphabet, qam_constellation, gray_bitmap, beta_star

def run_ber_point(precoder_fn , NT , NU , M , b , snr_db , n_trials , rng , P=1.0 , channel_fn = None) :
    """precoder_fn(s, H, C, NU, sigma_n2) -> unquantized x.
        channel_fn(NU, NT, rng) -> H; defaults to i.i.d. Rayleigh if None.
        "Strategy Pattern" to decouples how to run a fair simulation from what specific algorithm/channel you are testing.
    """

    from system_model import iid_Rayleigh_channel , Qb
    if channel_fn is None :
        channel_fn = iid_Rayleigh_channel

    levels , thresholds , C = gen_alphabet(b , P , NT)
    Sconst = qam_constellation(M)
    bitmap = gray_bitmap(M)
    sigma_n2 = P / (10** ( snr_db / 10 ))

    total_errs , total_bits = 0 , 0
    for _ in n_trials :
        H = channel_fn(NU , NT , rng)
        sym_idx = rng.integers(0 , M , NU)
        s = Sconst[sym_idx]
        true_bits = bitmap[sym_idx]

        x_raw = precoder_fn(s , H , C , NU , sigma_n2)
        x = Qb(x_raw, levels , thresholds)
        beta = beta_star(s, H , x , NU , sigma_n2)

        n = (rng.standard_normal(NU) + 1j * rng.standard_normal(NU)) * np.sqrt(sigma_n2 / 2)
        y = H @ x + n
        s_hat = beta * y
        det_idx = np.argmin(np.abs(s_hat[None , :] - Sconst[: , None]) , axis = 1)
        det_bits = bitmap[sym_idx]

        total_errs += np.sum(det_bits != true_bits)
        total_bits += true_bits.size

        return total_errs / total_bits

