import numpy as np

# ============================================================
#  FIR Filter Core Utilities
# ============================================================

def _sinc(x):
    return np.sinc(x)

def choose_window(A_s):
    """
    Selects the appropriate window type based on stopband attenuation.
    Returns (window_name, window_function, constant_for_N).
    """
    if A_s <= 50:
        return ("hamming", np.hamming, 3.3)
    elif A_s <= 74:
        return ("blackman", np.blackman, 5.5)
    else:
        return ("kaiser", np.kaiser, None)

def kaiser_beta(A_s):
    """Compute Kaiser beta parameter."""
    if A_s > 50:
        return 0.1102 * (A_s - 8.7)
    elif A_s >= 21:
        return 0
    else:
        return 0

def compute_N(A_s, tb_norm, win_name, const=None):
    """
    Compute filter order N (must be odd for Type-I FIR).
    """
    if win_name == "kaiser":
        N = int(np.ceil((A_s - 8) / (2.285 * 2 * np.pi * tb_norm)))
    else:
        if const is None:
            const = 3.3
        N = int(np.ceil(const / tb_norm))

    # Make N odd
    if N % 2 == 0:
        N += 1

    return N

def ideal_lowpass(fc, N):
    """Ideal lowpass impulse response."""
    M = (N - 1) // 2
    n = np.arange(N)
    return 2 * fc * _sinc(2 * fc * (n - M))


# ============================================================
#  MAIN FIR DESIGN (Used by GUI)
# ============================================================

def design_fir(filter_type, Fs, fcuts, A_s, trans_bw):
    """
    Main FIR design interface for GUI.
    Returns:
        h(n) coefficients (numpy array)
        meta-information dictionary
    """
    filter_type = filter_type.lower()
    tb_norm = trans_bw / Fs
    win_name, win_func, const = choose_window(A_s)

    N = compute_N(A_s, tb_norm, win_name, const)
    delta = trans_bw / (2 * Fs)

    if filter_type in ("low", "high"):
        fc = fcuts / Fs
        if filter_type == "low":
            fc_adj = fc + delta
            hd = ideal_lowpass(fc_adj, N)
        else:
            fc_adj = fc - delta
            lp = ideal_lowpass(fc_adj, N)
            hd = -lp
            hd[(N - 1)//2] += 1

    elif filter_type == "bandpass":
        f1 = fcuts[0] / Fs
        f2 = fcuts[1] / Fs
        f1_adj = f1 - delta
        f2_adj = f2 + delta
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        hd = lp2 - lp1

    elif filter_type == "bandstop":
        f1 = fcuts[0] / Fs
        f2 = fcuts[1] / Fs
        f1_adj = f1 - delta
        f2_adj = f2 + delta
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        band = lp2 - lp1
        hd = -band
        hd[(N - 1)//2] += 1

    # Apply window
    if win_name == "kaiser":
        w = np.kaiser(N, kaiser_beta(A_s))
    else:
        w = win_func(N)

    h = hd * w

    meta = {
        "N": N,
        "window": win_name,
        "A_s": A_s,
        "trans_bw": trans_bw
    }

    return h, meta


# ============================================================
#  TEXT-FILE BASED DESIGN (Used by Testing)
# ============================================================

def design_from_specs_file(spec_file):
    """
    Reads a test specification file and returns:
        indices = list of integers
        samples = list of floats
    Matching the exact expected output format.
    """
    with open(spec_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    params = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=")
            params[k.strip()] = v.strip()

    filter_type = params["FilterType"].lower()
    Fs = float(params["FS"])
    A_s = float(params["StopBandAttenuation"])
    trans = float(params["TransitionBand"])
    tb_norm = trans / Fs

    win_name, win_func, const = choose_window(A_s)
    N = compute_N(A_s, tb_norm, win_name, const)
    delta = trans / (2 * Fs)

    if "low" in filter_type:
        fc = float(params["FC"])
        fc_adj = (fc / Fs) + delta
        hd = ideal_lowpass(fc_adj, N)

    elif "high" in filter_type:
        fc = float(params["FC"])
        fc_adj = (fc / Fs) - delta
        lp = ideal_lowpass(fc_adj, N)
        hd = -lp
        hd[(N - 1)//2] += 1

    elif "bandpass" in filter_type:
        f1 = float(params["F1"])
        f2 = float(params["F2"])
        f1_adj = (f1 / Fs) - delta
        f2_adj = (f2 / Fs) + delta
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        hd = lp2 - lp1

    else:  # bandstop
        f1 = float(params["F1"])
        f2 = float(params["F2"])
        f1_adj = (f1 / Fs) - delta
        f2_adj = (f2 / Fs) + delta
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        band = lp2 - lp1
        hd = -band
        hd[(N - 1)//2] += 1

    # Window
    if win_name == "kaiser":
        w = np.kaiser(N, kaiser_beta(A_s))
    else:
        w = win_func(N)

    h = hd * w

    # Output must match test format: index then coefficient
    half = (N // 2)
    indices = list(range(-half, half + 1))
    samples = [float(v) for v in h]

    return indices, samples
