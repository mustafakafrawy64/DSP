import numpy as np
from tkinter import messagebox
import os

class Signal:
    def __init__(self, signal_type, is_periodic, samples, name=None):
        self.signal_type = signal_type
        self.is_periodic = is_periodic
        self.samples = samples
        self.name = name or "Signal"

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        signal_type = int(float(lines[0])) if len(lines) > 0 else 0
        is_periodic = int(float(lines[1])) if len(lines) > 1 else 0
        n = int(float(lines[2])) if len(lines) > 2 else 0
        samples = []
        for i in range(3, min(3+n, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 2:
                x = float(parts[0]); y = float(parts[1])
                samples.append([x, y])
        name = os.path.splitext(os.path.basename(filename))[0]
        return Signal(signal_type, is_periodic, samples, name=name)

    @staticmethod
    def from_arrays(x_vals, y_vals, signal_type=0, is_periodic=0, name=None):
        samples = [[float(x_vals[i]), float(y_vals[i])] for i in range(len(x_vals))]
        return Signal(signal_type, is_periodic, samples, name=name)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f"{self.signal_type}\n{self.is_periodic}\n{len(self.samples)}\n")
            for x, y in self.samples:
                f.write(f"{x} {y}\n")
        messagebox.showinfo("Saved", f"Signal saved to:\n{filename}")

    @staticmethod
    def _align_to_grid(target_x, sig):
        x = np.array([s[0] for s in sig.samples])
        y = np.array([s[1] for s in sig.samples])
        if len(x) == 0:
            return np.zeros_like(target_x)
        if len(x) == 1:
            return np.full_like(target_x, y[0], dtype=float)
        return np.interp(target_x, x, y)

    @staticmethod
    def combine_sum(signals, name="Sum"):
        if not signals: return None
        ref = signals[0]
        target_x = np.array([s[0] for s in ref.samples])
        total = np.zeros_like(target_x)
        for sig in signals:
            total += Signal._align_to_grid(target_x, sig)
        samples = [[float(target_x[i]), float(total[i])] for i in range(len(target_x))]
        return Signal(ref.signal_type, ref.is_periodic, samples, name=name)

    @staticmethod
    def combine_subtract(signals, name="Subtract"):
        if not signals: return None
        ref = signals[0]
        target_x = np.array([s[0] for s in ref.samples])
        total = Signal._align_to_grid(target_x, ref)
        for sig in signals[1:]:
            total -= Signal._align_to_grid(target_x, sig)
        samples = [[float(target_x[i]), float(total[i])] for i in range(len(target_x))]
        return Signal(ref.signal_type, ref.is_periodic, samples, name=name)

    def multiply_by_constant(self, c):
        return Signal(self.signal_type, self.is_periodic,
                      [[s[0], s[1] * c] for s in self.samples],
                      name=f"{self.name}*{c}")

    def square(self):
        return Signal(self.signal_type, self.is_periodic,
                      [[s[0], s[1] ** 2] for s in self.samples],
                      name=f"{self.name}^2")

    def normalize(self, min_range=0, max_range=1):
        y_vals = [s[1] for s in self.samples]
        if not y_vals:
            return Signal(self.signal_type, self.is_periodic, [], name=f"{self.name}_norm")
        y_min, y_max = min(y_vals), max(y_vals)
        if y_max == y_min:
            norm = [0 for _ in y_vals]
        else:
            norm = [(y - y_min) / (y_max - y_min) for y in y_vals]
        if (min_range, max_range) == (-1, 1):
            norm = [2 * n - 1 for n in norm]
        samples = [[self.samples[i][0], float(norm[i])] for i in range(len(norm))]
        return Signal(self.signal_type, self.is_periodic, samples, name=f"{self.name}_norm")

    def accumulate(self):
        acc = 0.0
        res = []
        for x, y in self.samples:
            acc += y
            res.append([x, acc])
        return Signal(self.signal_type, self.is_periodic, res, name=f"{self.name}_accum")
