# shift_signal.py
from signal_model import Signal
from typing import Tuple

def shift(sig: Signal, k: int, name: str = None) -> Signal:
    """
    Shift signal in time by k steps.
    - k > 0: delay (shift right) by k samples -> new x = old_x + k*dx
    - k < 0: advance (shift left)
    dx is inferred from first two x samples if available, else dx=1.
    """
    x_vals = [s[0] for s in sig.samples]
    y_vals = [float(s[1]) for s in sig.samples]
    if len(x_vals) < 2:
        dx = 1.0
    else:
        dx = x_vals[1] - x_vals[0]

    new_samples = [[float(x_vals[i] + k*dx), float(y_vals[i])] for i in range(len(x_vals))]
    return Signal(sig.signal_type, sig.is_periodic, new_samples, name=name or f"{sig.name}_shift_{k}")
