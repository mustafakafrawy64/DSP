import numpy as np
from tkinter import filedialog, messagebox

# Normalized sinc
def _sinc(x):
    return np.sinc(x)

# Choose window from stopband attenuation
def choose_window(A_s):
    if A_s <= 50:
        return ("hamming", np.hamming, 3.3)
    elif A_s <= 74:
        return ("blackman", np.blackman, 5.5)
    else:
        return ("kaiser", np.kaiser, None)

def _kaiser_beta(A_s):
    if A_s > 50:
        return 0.1102 * (A_s - 8.7)
    elif A_s >= 21:
        return 0.0
    else:
        return 0.0

# Compute N (odd)
def compute_N(A_s, trans_band_norm, window_name, approx_const=None):
    if trans_band_norm <= 0:
        raise ValueError("Transition band must be positive")

    if window_name == "kaiser":
        N = int(np.ceil((A_s - 8.0) / (2.285 * 2 * np.pi * trans_band_norm)))
    else:
        if approx_const is None:
            approx_const = 3.3
        N = int(np.ceil(approx_const / trans_band_norm))

    if N % 2 == 0:
        N += 1
    if N < 3:
        N = 3

    return N

# Ideal lowpass (Type I)
def ideal_lowpass(fc, N):
    M = (N - 1) // 2
    n = np.arange(N)
    return 2 * fc * _sinc(2 * fc * (n - M))

# Main FIR design
def design_fir(filter_type, Fs, fcuts, A_s, trans_bw):
    # Normalize
    if filter_type in ("low", "high"):
        f_c = float(fcuts) / Fs
        delta = trans_bw / (2 * Fs)

        if filter_type == "low":
            f_adj = f_c + delta
        else:
            f_adj = f_c - delta

        if not (0 < f_adj < 0.5):
            raise ValueError("Adjusted cutoff out of range")

    else:  # band filters
        f1 = fcuts[0] / Fs
        f2 = fcuts[1] / Fs
        delta = trans_bw / (2 * Fs)

        f1_adj = f1 - delta
        f2_adj = f2 + delta

        if not (0 < f1_adj < f2_adj < 0.5):
            raise ValueError("Adjusted band edges invalid")

    # Window choice
    win_name, win_func, approx_const = choose_window(A_s)
    trans_norm = trans_bw / Fs
    N = compute_N(A_s, trans_norm, win_name, approx_const)

    # Ideal responses
    if filter_type == "low":
        hd = ideal_lowpass(f_adj, N)

    elif filter_type == "high":
        lp = ideal_lowpass(f_adj, N)
        hd = -lp
        hd[(N - 1) // 2] += 1

    elif filter_type == "bandpass":
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        hd = lp2 - lp1

    elif filter_type == "bandstop":
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        band = lp2 - lp1
        hd = -band
        hd[(N - 1) // 2] += 1

    # Window
    if win_name == "kaiser":
        beta = _kaiser_beta(A_s)
        w = np.kaiser(N, beta)
    else:
        w = win_func(N)
        beta = None
