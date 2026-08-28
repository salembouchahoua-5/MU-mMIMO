import numpy as np
import itertools



def gen_alphabet(b, P, NT) :
    """Per-real-dimension quantizer levels/thresholds and the flattened
       complex DAC alphabet C_b (2^{2b} points), calibrated so that a
       uniformly-random alphabet point carries average power P/NT.

       This calibration is the standard M-QAM energy-normalization formula
       in disguise: with L = 2^b levels per real dimension, the L x L grid
       is exactly a square L^2-QAM constellation, scaled to the desired
       power budget.

       Returns
       -------
       levels      : (L,)  real quantizer output levels, eq (2), alpha=1
       thresholds  : (L-1,) real decision thresholds, eq (3)
       C           : (2^{2b},) complex DAC alphabet points c_i
    """

    L = 2**b
    l = np.arange(L)
    delta = np.sqrt( (6*P) / NT * (L**2 - 1) ) if L > 1 else delta = 1
    levels = delta * ( l - (L - 1) / 2.0)
    thresholds = delta * (np.arange(1,L) - L / 2.0)
    re , im = np.meshgrid(levels , levels)
    C = (re + 1j*im).flatten()

    return levels , thresholds , C

def quantize_real(x , levels , thresholds) :
    """Map a real array to the nearest level via eq (4)."""
    return levels[np.searchsorted(x, thresholds , side = "right")]

def Qb(x, levels , thresholds ) :
    """Element-wise complex quantization: quantize real and imaginary
        parts separately, eq (4).
    """
    return (quantize_real(x.reals , levels , thresholds ) )+ 1j * (quantize_real(x.imag, levels , thresholds))

def qam_constellation(M) :
    """Standard square M-QAM constellation, unit average energy."""
    m = int(round(np.sqrt(M)))
    assert m*m == M
    levels = np.arange(m) - ( m - 1 ) / 2.0
    re , im = np.meshgrid(m,m)
    C = (re + 1j * im).flatten()
    return C / np.sqrt(np.mean(np.abs(C)**2))


def gray_bitmap(M):
    """A consistent (not necessarily literature-standard Gray) bit mapping
    for a size-M constellation, for BER counting. Any fixed, consistent
    mapping is valid for reproducing *trends*; only exact literature BER
    values require matching the paper's own (unspecified) mapping.
    This uses standard binary code ordering rather than a geometric Gray code.
    While true Gray coding minimizes Bit Error Rate (BER) by ensuring adjacent
    points differ by only 1 bit, any fixed 1-to-1 mapping is valid for comparing
    relative performance trends across algorithms.
    """
    n_bits = int(np.log2(M))
    return np.array([[int(b) for b in format(i, f'0{n_bits}b')] for i in range(M)])

# 3. CHANNEL MODELS  (Sec. V, eqs 49-51)

def iid_Rayleigh_channel(NU , NT , rng ) :
    """H with i.i.d. CN(0,1) entries."""
    return (rng.standard_normal(NU , NT) + 1j * rng.standard_normal(NU , NT) )/np.sqrt(2)

def apply_csi_error(H , tau) :
    """Gauss-Markov CSI error model, eq (49): Hhat = sqrt(1-tau^2) H + tau E,
    with E ~ i.i.d. CN(0,1) entries, tau in [0,1].
    In a system simulation, use the output `Hhat` to calculate your precoding
    or combining weights, but apply those weights to the input `H` to calculate
    the actual physical received signal and SNR."""
    NU , NT = H.shape
    E = iid_Rayleigh_channel(NU, NT)

    return np.sqrt(1 - tau**2) * H + tau * E

def jakes_correlation_matrix(N , d_over_lambda) :
    """Transmit-side spatial correlation matrix for a ULA under isotropic
        scattering, eq (51): [R_tx]_{p,q} = J0(2*pi*d/lambda*|p-q|)."""
    from scipy.special import j0
    idx = np.arange(N)
    diff = np.abs(idx[: , None] - idx[None , :])

    return j0(2 * np.pi * d_over_lambda * diff)

def Kronecker_correlated_channel(NU , NT , d_over_lambda , rng) :

    Rtx = jakes_correlation_matrix(NT , d_over_lambda)
    Rtx_half = np.linalg.cholesky(Rtx + 1e-10*np.eye(NT))

    H = iid_Rayleigh_channel(NU , NT , rng)

    return H @ Rtx_half.T

# 4. CORE PRECODING MATH  (Sec. II-C, III; eqs 6, 7, 9)

def beta_star(s, H , x , NU , sigma_n2) :
    """Closed-form optimal receiver scaling, eq (9)."""

    Hx = H @ x
    num = np.real(np.vdot(s,Hx))
    den = np.linalg.norm(Hx)**2 + NU * sigma_n2

    return num / den

def objective(s , beta , H , x , NU , sigma_n2 ) :
    """The receive-side MSE objective, eq (6)/(7)."""

    return np.linalg.norm(s - np.vdot(beta , H @ x)) ** 2 + beta**2 * NU * sigma_n2
