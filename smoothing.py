# smoothing.py
import numpy as np
from signal_model import Signal

def moving_average(sig: Signal, M: int, name: str = None) -> Signal:
    """
    Compute moving average (trailing window) over M samples.
    - sig: input Signal
    - M: number of points included in averaging (integer >=1)
    Returns new Signal with same x indices and averaged y values.
    """
    if M <= 0:
        raise ValueError("M must be >= 1")
    x_vals = [s[0] for s in sig.samples]
    y_vals = [float(s[1]) for s in sig.samples]
    N = len(y_vals)
    if N == 0:
        return Signal(sig.signal_type, sig.is_periodic, [], name=name or f"{sig.name}_smooth")

    y = np.array(y_vals, dtype=float)
    # trailing moving average using cumulative sum for efficiency
    csum = np.concatenate([[0.0], np.cumsum(y)])
    smoothed = []
    for i in range(N):
        start = max(0, i - M + 1)
        window_sum = csum[i+1] - csum[start]
        window_len = i - start + 1
        smoothed.append(float(window_sum / window_len))

    samples = [[float(x_vals[i]), float(smoothed[i])] for i in range(N)]
    return Signal(sig.signal_type, sig.is_periodic, samples, name=name or f"{sig.name}_smooth")
