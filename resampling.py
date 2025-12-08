
import numpy as np
from signal_model import Signal
from scipy.signal import firwin, lfilter

# ==========================================================
# FIR LOW-PASS FILTER DESIGN
# ==========================================================
def design_lowpass_fir(Fs, Fc, As, TB):
    N = int(np.ceil(4 / (TB / Fs)))
    if N % 2 == 0:
        N += 1
    h = firwin(N, Fc / (Fs / 2))
    return h

# ==========================================================
# UPSAMPLING
# ==========================================================
def upsample_signal(sig, L):
    if L <= 1:
        return sig
    y = np.array([s[1] for s in sig.samples])
    up_y = np.zeros(len(y) * L)
    up_y[::L] = y
    return Signal.from_arrays(list(range(len(up_y))), up_y,
                              signal_type=sig.signal_type, is_periodic=sig.is_periodic,
                              name=f"{sig.name}_up{L}")

# ==========================================================
# DOWNSAMPLING
# ==========================================================
def downsample_signal(sig, M):
    if M <= 1:
        return sig
    y = np.array([s[1] for s in sig.samples])
    down_y = y[::M]
    return Signal.from_arrays(list(range(len(down_y))), down_y,
                              signal_type=sig.signal_type, is_periodic=sig.is_periodic,
                              name=f"{sig.name}_down{M}")

# ==========================================================
# FIR FILTER
# ==========================================================
def apply_fir_filter(sig, h):
    y = np.array([s[1] for s in sig.samples])
    filtered_y = lfilter(h, 1.0, y)
    delay = (len(h)-1)//2
    filtered_y = np.roll(filtered_y, -delay)
    filtered_y[-delay:] = 0
    return Signal.from_arrays(list(range(len(filtered_y))), filtered_y,
                              signal_type=sig.signal_type, is_periodic=sig.is_periodic,
                              name=f"{sig.name}_filt")

# ==========================================================
# RESAMPLING
# ==========================================================
def resample_signal(sig, M, L,
                    InputFS=8000,
                    InputCutOffFrequency=1500,
                    InputStopBandAttenuation=50,
                    InputTransitionBand=500,
                    start_index=-26):
    if M == 0 and L == 0:
        raise ValueError("Both M and L cannot be zero.")

    h = design_lowpass_fir(InputFS, InputCutOffFrequency, InputStopBandAttenuation, InputTransitionBand)
    result = sig

    if M == 0 and L > 0:
        result = upsample_signal(sig, L)
        result = apply_fir_filter(result, h)
    elif M > 0 and L == 0:
        result = apply_fir_filter(sig, h)
        result = downsample_signal(result, M)
    elif M > 0 and L > 0:
        result = upsample_signal(sig, L)
        result = apply_fir_filter(result, h)
        result = downsample_signal(result, M)

    # Ensure integer indexing starting from start_index
    y_len = len(result.samples)
    int_x = [start_index + i for i in range(y_len)]
    y_vals = [s[1] for s in result.samples]

    return Signal.from_arrays(int_x, y_vals, signal_type=result.signal_type,
                              is_periodic=result.is_periodic, name=result.name)
