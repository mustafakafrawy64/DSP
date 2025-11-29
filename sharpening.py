# sharpening.py
from signal_model import Signal
from typing import Literal

DerivType = Literal["first", "second"]

def derivative(sig: Signal, kind: DerivType = "first", name: str = None) -> Signal:
    """
    Compute derivative:
      - kind == "first": y[n] = x[n] - x[n-1], with y[0] = x[0] (or 0 if you prefer)
      - kind == "second": y[n] = x[n+1] - 2*x[n] + x[n-1], with boundary handling (assume x out-of-range = 0)
    """
    x_vals = [s[0] for s in sig.samples]
    y_vals = [float(s[1]) for s in sig.samples]
    N = len(y_vals)
    if N == 0:
        return Signal(sig.signal_type, sig.is_periodic, [], name=name or f"{sig.name}_deriv")

    out = [0.0] * N
    if kind == "first":
        # y[0] choose x[0] - x[-1] but there's no x[-1]; choose y[0] = x[0]
        out[0] = float(y_vals[0])
        for n in range(1, N):
            out[n] = float(y_vals[n] - y_vals[n-1])
    elif kind == "second":
        # Use zero padding for out-of-range neighbors
        for n in range(N):
            xm1 = y_vals[n-1] if n-1 >= 0 else 0.0
            xp1 = y_vals[n+1] if n+1 < N else 0.0
            out[n] = float(xp1 - 2.0*y_vals[n] + xm1)
    else:
        raise ValueError("kind must be 'first' or 'second'")

    samples = [[float(x_vals[i]), float(out[i])] for i in range(N)]
    suffix = "firstDer" if kind=="first" else "secondDer"
    return Signal(sig.signal_type, sig.is_periodic, samples, name=name or f"{sig.name}_{suffix}")
