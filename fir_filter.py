# fir_filter.py
# Unified FIR design + test comparison utilities.
import numpy as np
import re

def _sinc(x):
    return np.sinc(x)

def choose_window(A_s):
    if A_s <= 50:
        return ("hamming", np.hamming, 3.3)
    elif A_s <= 74:
        return ("blackman", np.blackman, 5.5)
    else:
        return ("kaiser", np.kaiser, None)

def kaiser_beta(A_s):
    if A_s > 50:
        return 0.1102 * (A_s - 8.7)
    elif A_s >= 21:
        return 0.0
    else:
        return 0.0

def compute_N(A_s, tb_norm, win_name, const=None):
    if tb_norm <= 0:
        raise ValueError("Transition band must be positive")

    if win_name == "kaiser":
        N_float = (A_s - 8.0) / (2.285 * 2.0 * np.pi * tb_norm)
        N = int(np.ceil(N_float))
    else:
        if const is None:
            const = 3.3
        N = int(np.ceil(const / tb_norm))

    if N % 2 == 0:
        N += 1
    if N < 3:
        N = 3
    return N

def ideal_lowpass(fc, N):
    M = (N - 1) // 2
    n = np.arange(N)
    return 2.0 * fc * _sinc(2.0 * fc * (n - M))

def design_fir(filter_type, Fs, fcuts, A_s, trans_bw):
    ft = filter_type.lower()
    win_name, win_func, const = choose_window(A_s)
    tb_norm = trans_bw / float(Fs)
    N = compute_N(A_s, tb_norm, win_name, const)
    delta = trans_bw / (2.0 * Fs)

    if ft in ("low", "high"):
        fc = float(fcuts) / float(Fs)
        if ft == "low":
            fc_adj = fc + delta
            hd = ideal_lowpass(fc_adj, N)
        else:
            fc_adj = fc - delta
            lp = ideal_lowpass(fc_adj, N)
            hd = -lp
            hd[(N - 1)//2] += 1.0

    elif ft == "bandpass":
        f1 = float(fcuts[0]) / float(Fs)
        f2 = float(fcuts[1]) / float(Fs)
        f1_adj = f1 - delta
        f2_adj = f2 + delta
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        hd = lp2 - lp1

    elif ft == "bandstop":
        f1 = float(fcuts[0]) / float(Fs)
        f2 = float(fcuts[1]) / float(Fs)
        f1_adj = f1 - delta
        f2_adj = f2 + delta
        lp1 = ideal_lowpass(f1_adj, N)
        lp2 = ideal_lowpass(f2_adj, N)
        band = lp2 - lp1
        hd = -band
        hd[(N - 1)//2] += 1.0

    else:
        raise ValueError("Unsupported filter type")

    if win_name == "kaiser":
        beta = kaiser_beta(A_s)
        w = np.kaiser(N, beta)
    else:
        w = win_func(N)

    h = hd * w
    meta = {"N": int(N), "window": win_name, "A_s": A_s, "trans_bw": trans_bw}
    return h.astype(float), meta

def design_from_specs_file(spec_file):
    params = {}
    with open(spec_file, "r") as f:
        for raw in f:
            if "=" not in raw:
                continue
            k,v = raw.split("=",1)
            params[k.strip().lower()] = v.strip()

    # Normalize filter type (remove spaces and convert common phrases)
    raw_ftype = params["filtertype"].lower().replace(" ", "")

    # map synonyms
    if raw_ftype in ("lowpass", "lpf"):
        ftype = "low"
    elif raw_ftype in ("highpass", "hpf"):
        ftype = "high"
    elif raw_ftype in ("bandpass", "bpf"):
        ftype = "bandpass"
    elif raw_ftype in ("bandstop", "bsf", "notch"):
        ftype = "bandstop"
    else:
        raise ValueError(f"Unsupported filter type: {params['filtertype']}")

    Fs = float(params["fs"])
    A_s = float(params["stopbandattenuation"])
    trans = float(params["transitionband"])

    if "low" in ftype or "high" in ftype:
        fcuts = float(params["fc"])
    else:
        fcuts = (float(params["f1"]), float(params["f2"]))

    h, meta = design_fir(ftype, Fs, fcuts, A_s, trans)
    N = meta["N"]
    half = N//2
    indices = list(range(-half, half+1))
    samples = [float(x) for x in h]
    return indices, samples

def _parse_expected_file(expected_file):
    idxs = []
    vals = []
    with open(expected_file, "r") as f:
        for raw in f:
            parts = raw.split()
            if len(parts)>=2:
                try:
                    idxs.append(int(parts[0]))
                    vals.append(float(parts[1]))
                except:
                    pass
    return idxs, vals

def compare_with_reference(indices, samples, expected_file, tol=0.01):
    exp_i, exp_v = _parse_expected_file(expected_file)
    if len(exp_v)==0:
        return False, "Expected file empty"
    if len(exp_v)!=len(samples):
        return False, "Length mismatch"
    for a,b in zip(indices,exp_i):
        if a!=b:
            return False, f"Index mismatch {a} != {b}"
    for a,b in zip(samples,exp_v):
        if abs(a-b)>tol:
            return False, f"Value mismatch {a} != {b}"
    return True, "PASS"
