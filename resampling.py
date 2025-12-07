import numpy as np
from signal_model import Signal
from fir_filter import design_fir, apply_filter_to_signal

# ---------------------------------------------------------
#   BASIC RESAMPLING OPERATIONS
# ---------------------------------------------------------

def upsample(sig, L):
    samples = sig.samples
    x = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples])

    new_x = np.arange(0, len(y) * L)
    new_y = np.zeros(len(y) * L)

    new_y[::L] = y

    return Signal(sig.signal_type, sig.is_periodic,
                  [[float(new_x[i]), float(new_y[i])] for i in range(len(new_x))],
                  name=f"{sig.name}_up{L}")


def downsample(sig, M):
    samples = sig.samples
    x = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples])

    new_x = x[::M]
    new_y = y[::M]

    return Signal(sig.signal_type, sig.is_periodic,
                  [[float(new_x[i]), float(new_y[i])] for i in range(len(new_x))],
                  name=f"{sig.name}_down{M}")


# ---------------------------------------------------------
#   RESAMPLING CORE FUNCTION
# ---------------------------------------------------------
def resample_signal(sig, M, L):
    """
    Handles all four required cases:

    1) M=0, L>0       → Upsample → Lowpass filter
    2) M>0, L=0       → Lowpass filter → Downsample
    3) M>0, L>0       → Upsample → Lowpass filter → Downsample
    4) M=0, L=0       → ERROR
    """

    if M == 0 and L == 0:
        raise ValueError("Both M and L cannot be zero.")

    # Filter specs (fixed)
    filter_type = "low"
    Fs = 8000
    As = 50
    Fc = 1500
    TB = 500

    # Step 1 → Upsample (if L > 0)
    if L > 0:
        sig_up = upsample(sig, L)
    else:
        sig_up = sig

    # Step 2 → Low-pass filter
    h, meta = design_fir(filter_type, Fs, Fc, As, TB)
    sig_filtered = apply_filter_to_signal(sig_up, h, name=f"{sig_up.name}_filt")

    # Step 3 → Downsample (if M > 0)
    if M > 0:
        sig_final = downsample(sig_filtered, M)
    else:
        sig_final = sig_filtered

    return sig_final
