# folding.py
from signal_model import Signal

def fold(sig: Signal, name: str = None) -> Signal:
    """
    Fold (time-reverse) the signal: produce samples corresponding to y(-n).
    Implementation: new_x = -old_x; new_y = old_y; then sort by new_x ascending.
    """
    new_samples = [[float(-s[0]), float(s[1])] for s in sig.samples]
    # sort by x to maintain increasing x order for plotting
    new_samples_sorted = sorted(new_samples, key=lambda p: p[0])
    return Signal(sig.signal_type, sig.is_periodic, new_samples_sorted, name=name or f"{sig.name}_fold")
