# fold_shift.py
from folding import fold
from shift_signal import shift
from signal_model import Signal

def fold_and_shift(sig: Signal, k: int, name: str = None) -> Signal:
    """
    Fold (time reverse) then shift by k steps.
    """
    folded = fold(sig, name=(name or f"{sig.name}_fold"))
    shifted = shift(folded, k, name=name or f"{sig.name}_fold_shift_{k}")
    return shifted
