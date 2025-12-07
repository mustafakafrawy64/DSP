import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Project-specific imports (assumes these modules exist in the project folder)
try:
    from signal_model import Signal
except Exception as e:
    raise ImportError("signal_model.py is required and must be in the same folder as gui.py") from e

try:
    from signal_generator import GenerateSignal
except Exception:
    # Provide a fallback stub to avoid crashes when generate is not present.
    def GenerateSignal(sig_type, A, F, Fs, phase):
        # produce a simple sampled sinusoid
        t = np.arange(0, 1, 1.0 / Fs)
        if sig_type == "sin":
            y = A * np.sin(2 * np.pi * F * t + phase)
        else:
            y = A * np.cos(2 * np.pi * F * t + phase)
        return list(t), list(y)


try:
    from fir_filter import design_fir, save_coefficients, apply_filter_to_signal
except Exception:
    def design_fir(*a, **k):
        raise ImportError('fir_filter.py missing')
    def save_coefficients(*a, **k):
        raise ImportError('fir_filter.py missing')
    def apply_filter_to_signal(*a, **k):
        raise ImportError('fir_filter.py missing')

try:
    from utils import unique_name
except Exception:
    # Minimal fallback
    def unique_name(base, existing):
        name = base
        i = 1
        while name in existing:
            name = f"{base}_{i}"
            i += 1
        return name

# Quantization / FFT helper imports (optional)
try:
    from quantization import quantize_signal
except Exception:
    def quantize_signal(sig, bits):
        # simple uniform quantization fallback
        y = [s[1] for s in sig.samples]
        mn, mx = min(y), max(y)
        if mx == mn:
            return Signal(sig.signal_type, sig.is_periodic, [[s[0], 0.0] for s in sig.samples], name=(sig.name + "_Q"))
        levels = 2**bits
        step = (mx - mn) / (levels - 1)
        q = [round((v - mn) / step) * step + mn for v in y]
        samples = [[sig.samples[i][0], float(q[i])] for i in range(len(q))]
        return Signal(sig.signal_type, sig.is_periodic, samples, name=(sig.name + f"_Q{bits}"))

try:
    from fft_ifft_handler import fft_ifft_handler
except Exception:
    def fft_ifft_handler(sigs):
        messagebox.showinfo("FFT/IFFT", "FFT handler not available (fft_ifft_handler.py missing).")

# Fourier transform functions (optional)
try:
    from Fourier_transform import (
        dft, idft, get_dominant_frequencies,
        test_reconstructed_signal, run_comparison_test
    )
except Exception:
    # Provide stubs so the GUI still loads even if Fourier_transform.py is missing.
    def dft(y, fs):
        return None, None, None, None, None

    def idft(c):
        return []

    def get_dominant_frequencies(f, a):
        return []

    def test_reconstructed_signal(sig, file):
        messagebox.showwarning("Missing", "Fourier transform test function not available.")

    def run_comparison_test(file, amps, phases):
        messagebox.showwarning("Missing", "DFT comparison test not available.")

# New feature modules (these files should exist alongside gui.py)
try:
    from smoothing import moving_average
except Exception:
    def moving_average(sig, M, name=None):
        raise ImportError("smoothing.py missing")

try:
    from sharpening import derivative
except Exception:
    def derivative(sig, kind="first", name=None):
        raise ImportError("sharpening.py missing")

try:
    from shift_signal import shift
except Exception:
    def shift(sig, k, name=None):
        raise ImportError("shift_signal.py missing")

try:
    from folding import fold
except Exception:
    def fold(sig, name=None):
        raise ImportError("folding.py missing")

try:
    from fold_shift import fold_and_shift
except Exception:
    def fold_and_shift(sig, k, name=None):
        raise ImportError("fold_shift.py missing")


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")
        self.root.geometry("1366x650")

        self.signals = []
        self.result_signal = None

        # --- FT DATA STORAGE ---
        self.freq_components = None
        self.frequencies = None
        self.amplitudes_norm = None
        self.amplitudes_unnorm = None
        self.phases = None
        self.current_fs = None
        self.selected_signal_for_ft = None
        # --- END FT DATA STORAGE ---

        # === Left panel ===
        left_frame = tk.Frame(self.root, width=220)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        tk.Label(left_frame, text="Signals (select one or many):").pack(anchor="w")
        self.listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED, width=30, height=25)
        self.listbox.pack(fill=tk.Y, expand=False)

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=6)

        tk.Button(btn_frame, text="Load...", command=self.load_signals).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Remove", command=self.remove_selected).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Save", command=self.save_selected).pack(side=tk.LEFT, padx=2)

        # === Plot area ===
        self.fig, (self.ax_cont, self.ax_disc) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # === Zoom controls ===
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, anchor="ne", padx=6, pady=4)
        tk.Button(control_frame, text="Zoom In", command=lambda: self.zoom(0.5)).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Zoom Out", command=lambda: self.zoom(2.0)).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Plot Selected", command=self.plot_selected).pack(side=tk.LEFT, padx=6)

        # === Menu bar ===
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # --- File menu ---
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Signals...", command=self.load_signals)
        file_menu.add_command(label="Generate Signal...", command=self.generate_signal_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Save Selected...", command=self.save_selected)
        menubar.add_cascade(label="File", menu=file_menu)

        # --- View menu ---
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Plot Selected (Time Domain)", command=self.plot_selected)
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=lambda: self.zoom(0.5))
        view_menu.add_command(label="Zoom Out", command=lambda: self.zoom(2.0))
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        menubar.add_cascade(label="View", menu=view_menu)

        # --- Operations menu ---
        ops_menu = tk.Menu(menubar, tearoff=0)
        ops_menu.add_command(label="Add (Sum selected)", command=self.add_selected_signals)
        ops_menu.add_command(label="Subtract (first - rest)", command=self.subtract_selected_signals)
        ops_menu.add_separator()
        ops_menu.add_command(label="Multiply by constant", command=self.multiply_selected)
        ops_menu.add_command(label="Square", command=self.square_selected)
        ops_menu.add_command(label="Normalize", command=self.normalize_selected)
        ops_menu.add_command(label="Accumulate", command=self.accumulate_selected)
        ops_menu.add_command(label="Convolve (2 signals)", command=self.convolve_selected_signals)
        ops_menu.add_separator()
        ops_menu.add_command(label="Correlation (Auto / Cross)", command=self.correlation_dialog)
        ops_menu.add_command(label="Periodic Cross-Correlation", command=self.periodic_cross_correlation)
        ops_menu.add_command(label="Time Delay Analysis", command=self.time_delay_analysis)

        # === NEW: Time-domain helpers (smoothing, sharpening, shifting, folding) ===
        ops_menu.add_separator()
        ops_menu.add_command(label="Smoothing (Moving Average)", command=self.smoothing_dialog)
        ops_menu.add_command(label="Sharpening (Derivatives)", command=self.sharpening_dialog)
        ops_menu.add_command(label="Delay/Advance Signal by k", command=self.shift_dialog)
        ops_menu.add_command(label="Fold (Time Reverse) Signal", command=self.fold_dialog)
        ops_menu.add_command(label="Fold then Delay/Advance", command=self.fold_shift_dialog)
        ops_menu.add_separator()
        ops_menu.add_command(label="Filter (FIR, Window Method)", command=self.filtering_dialog)
        ops_menu.add_command(label="Resampling (M,L)", command=self.resampling_dialog)



        # Quantization
        ops_menu.add_separator()
        ops_menu.add_command(label="Quantize Signal", command=self.quantize_selected_signal)

        menubar.add_cascade(label="Operations", menu=ops_menu)

        # --- FREQUENCY DOMAIN MENU ---
        ft_menu = tk.Menu(menubar, tearoff=0)
        ft_menu.add_command(label="Apply Fourier Transform (DFT)", command=self.apply_dft)
        ft_menu.add_command(label="Show Dominant Frequencies", command=self.show_dominant_frequencies)
        ft_menu.add_separator()
        ft_menu.add_command(label="FFT / IFFT", command=lambda: fft_ifft_handler(self._get_selected_signals()))
        ft_menu.add_separator()
        ft_menu.add_command(label="Remove DC Component", command=self.remove_dc)
        ft_menu.add_command(label="Modify Components...", command=self.modify_components_dialog)
        ft_menu.add_separator()
        ft_menu.add_command(label="Reconstruct Signal (IDFT)", command=self.apply_idft)
        ft_menu.add_separator()
        ft_menu.add_command(label="Remove DC (Time Domain)", command=self.remove_dc_time_domain)
        ft_menu.add_separator()
        ft_menu.add_command(label="Test Signal vs. File... (Time-Domain)", command=self.test_reconstruction_dialog)
        ft_menu.add_command(label="Test DFT Output vs. File... (Freq-Domain)", command=self.test_dft_output_dialog)

        menubar.add_cascade(label="Frequency Domain", menu=ft_menu)

    # =======================
    # SIGNAL MANAGEMENT
    # =======================
    def load_signals(self):
        files = filedialog.askopenfilenames(
            title="Load signals",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not files:
            return
        for file in files:
            try:
                sig = Signal.from_file(file)
                sig.name = unique_name(sig.name, [s.name for s in self.signals])
                self.signals.append(sig)
            except Exception as e:
                messagebox.showerror("Error loading signal", f"Could not load {file}:\n{e}")
        self._refresh_listbox()

    def remove_selected(self):
        selected = list(self.listbox.curselection())
        selected.reverse()
        for i in selected:
            del self.signals[i]
        self._refresh_listbox()
        # Clear FT data if the signal being analyzed was removed
        self.freq_components = None

    def save_selected(self):
        selected = self._get_selected_signals()
        if not selected:
            return
        for sig in selected:
            file = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt")],
                                                initialfile=f"{sig.name}.txt")
            if file:
                try:
                    sig.save_to_file(file)
                except Exception as e:
                    messagebox.showerror("Save error", f"Failed to save {sig.name}:\n{e}")

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for sig in self.signals:
            self.listbox.insert(tk.END, sig.name)

    def _get_selected_signals(self):
        indices = self.listbox.curselection()
        return [self.signals[i] for i in indices]

    # =======================
    # PLOTTING
    # =======================
    def plot_selected(self):
        """Plots the selected time-domain signals."""
        sels = self._get_selected_signals()
        self.ax_cont.clear()
        self.ax_disc.clear()

        for sig in sels:
            x = np.array([s[0] for s in sig.samples])
            y = np.array([s[1] for s in sig.samples])

            if len(x) == 0 or len(y) == 0:
                continue

            # Continuous plot
            if len(x) > 1 and x.min() != x.max():
                dense_x = np.linspace(x.min(), x.max(), max(200, len(x) * 10))
                dense_y = np.interp(dense_x, x, y)
                self.ax_cont.plot(dense_x, dense_y, label=sig.name)
            else:
                self.ax_cont.plot(x, y, marker='o', linestyle='none', label=sig.name)

            # Discrete plot
            markerline, stemlines, baseline = self.ax_disc.stem(x, y, label=sig.name)
            markerline.set_marker('o')
            markerline.set_markersize(6)

        self.ax_cont.set_title("Continuous (Time Domain)")
        self.ax_disc.set_title("Discrete (Time Domain)")
        self.ax_disc.set_xlabel("Time / Index")
        self.ax_cont.legend()
        self.ax_disc.legend()
        self.canvas.draw()

    # =======================
    # ZOOM
    # =======================
    def zoom(self, factor):
        for ax in (self.ax_cont, self.ax_disc):
            xlim = ax.get_xlim()
            xcenter = (xlim[0] + xlim[1]) / 2
            xhalf = (xlim[1] - xlim[0]) / 2 * factor
            ax.set_xlim(xcenter - xhalf, xcenter + xhalf)
        self.canvas.draw()

    def reset_zoom(self):
        self.ax_cont.autoscale()
        self.ax_disc.autoscale()
        self.canvas.draw()

    # =======================
    # SIGNAL GENERATION
    # =======================
    def generate_signal_dialog(self):
        sig_type = simpledialog.askstring("Signal Type", "Enter signal type (sin or cos):")
        if sig_type not in ("sin", "cos"):
            messagebox.showerror("Invalid type", "Type must be 'sin' or 'cos'")
            return

        try:
            A = float(simpledialog.askstring("Amplitude", "Enter amplitude (e.g. 1):"))
            F = float(simpledialog.askstring("Analog Frequency", "Enter analog frequency (Hz):"))
            Fs = float(simpledialog.askstring("Sampling Frequency", "Enter sampling frequency (Hz):"))
            phase = float(simpledialog.askstring("Phase Shift", "Enter phase shift (radians):"))
        except (TypeError, ValueError):
            messagebox.showerror("Error", "Invalid numeric input")
            return

        if Fs < 2 * F:
            messagebox.showerror("Error", "Sampling frequency must be ≥ 2× analog frequency")
            return

        x, y = GenerateSignal(sig_type, A, F, Fs, phase)
        sig = Signal.from_arrays(x, y, signal_type=0, is_periodic=1,
                                 name=unique_name(sig_type, [s.name for s in self.signals]))
        self.signals.append(sig)
        self._refresh_listbox()

    # =======================
    # OPERATIONS
    # =======================
    def add_selected_signals(self):
        sels = self._get_selected_signals()
        if len(sels) < 2:
            messagebox.showwarning("Warning", "Select at least two signals to add.")
            return
        result = Signal.combine_sum(sels, name=unique_name("Sum", [s.name for s in self.signals]))
        self.signals.append(result)
        self._refresh_listbox()

    def subtract_selected_signals(self):
        sels = self._get_selected_signals()
        if len(sels) < 2:
            messagebox.showwarning("Warning", "Select at least two signals to subtract.")
            return
        result = Signal.combine_subtract(sels, name=unique_name("Sub", [s.name for s in self.signals]))
        self.signals.append(result)
        self._refresh_listbox()

    def multiply_selected(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal to multiply.")
            return
        try:
            c = float(simpledialog.askstring("Multiply", "Enter constant:"))
        except:
            return
        result = sels[0].multiply_by_constant(c)
        result.name = unique_name(result.name, [s.name for s in self.signals])
        self.signals.append(result)
        self._refresh_listbox()

    def square_selected(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal to square.")
            return
        result = sels[0].square()
        result.name = unique_name(result.name, [s.name for s in self.signals])
        self.signals.append(result)
        self._refresh_listbox()

    def normalize_selected(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal to normalize.")
            return
        result = sels[0].normalize()
        result.name = unique_name(result.name, [s.name for s in self.signals])
        self.signals.append(result)
        self._refresh_listbox()

    def accumulate_selected(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal to accumulate.")
            return
        result = sels[0].accumulate()
        result.name = unique_name(result.name, [s.name for s in self.signals])
        self.signals.append(result)
        self._refresh_listbox()

    # -----------------------
    # Convolution
    # -----------------------
    def convolve_selected_signals(self):
        sels = self._get_selected_signals()
        if len(sels) != 2:
            messagebox.showwarning("Warning", "Select exactly two signals to convolve.")
            return
        sig1, sig2 = sels
        try:
            result = sig1.convolve(sig2)
            result.name = unique_name(f"{sig1.name}_conv_{sig2.name}", [s.name for s in self.signals])
            self.signals.append(result)
            self._refresh_listbox()
            messagebox.showinfo("Convolution", f"Created '{result.name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Convolution failed:\n{e}")

    # -----------------------
    # Correlation
    # -----------------------
    def correlation_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) == 1:
            try:
                result = sels[0].auto_correlation()
                result.name = unique_name(f"{sels[0].name}_autoCorr", [s.name for s in self.signals])
                self.signals.append(result)
                self._refresh_listbox()
                messagebox.showinfo("Auto-Correlation", f"Created '{result.name}'.")
            except Exception as e:
                messagebox.showerror("Error", f"Auto-correlation failed:\n{e}")

        elif len(sels) == 2:
            try:
                result = sels[0].cross_correlation(sels[1])
                result.name = unique_name(f"{sels[0].name}_xcorr_{sels[1].name}", [s.name for s in self.signals])
                self.signals.append(result)
                self._refresh_listbox()
                messagebox.showinfo("Cross-Correlation", f"Created '{result.name}'.")
            except Exception as e:
                messagebox.showerror("Error", f"Cross-correlation failed:\n{e}")

        else:
            messagebox.showwarning("Warning", "Select one or two signals.")

    # -----------------------
    # Periodic Cross-Correlation
    # -----------------------
    def periodic_cross_correlation(self):
        sels = self._get_selected_signals()
        if len(sels) != 2:
            messagebox.showwarning("Warning", "Select exactly two signals.")
            return
        try:
            result = sels[0].periodic_cross_correlation(sels[1])
            result.name = unique_name(f"{sels[0].name}_pcorr_{sels[1].name}", [s.name for s in self.signals])
            self.signals.append(result)
            self._refresh_listbox()
            messagebox.showinfo("Periodic CC", f"Created '{result.name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Periodic CC failed:\n{e}")

    # -----------------------
    # Time Delay Analysis
    # -----------------------
    def time_delay_analysis(self):
        sels = self._get_selected_signals()
        if len(sels) != 2:
            messagebox.showwarning("Warning", "Select two signals.")
            return
        try:
            delay = sels[0].time_delay(sels[1])
            messagebox.showinfo("Time Delay", f"Estimated delay: {delay}")
        except Exception as e:
            messagebox.showerror("Error", f"Time delay failed:\n{e}")

    # -----------------------
    # Smoothing
    # -----------------------
    def smoothing_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal to smooth.")
            return
        sig = sels[0]
        try:
            M_str = simpledialog.askstring("Moving Average", "Enter number of points included in averaging (integer >=1):")
            if M_str is None:
                return
            M = int(M_str)
            if M < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Invalid integer for M.")
            return
        try:
            new_sig = moving_average(sig, M, name=unique_name(f"{sig.name}_smooth", [s.name for s in self.signals]))
        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed:\n{e}")
            return
        self.signals.append(new_sig)
        self._refresh_listbox()
        messagebox.showinfo("Done", f"Smoothing applied with M={M}. New signal '{new_sig.name}' created.")

    # -----------------------
    # Sharpening (Derivatives)
    # -----------------------
    def sharpening_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal.")
            return
        sig = sels[0]
        choice = simpledialog.askstring("Derivative", "Enter 'first' or 'second' for derivative type:")
        if choice not in ("first", "second"):
            messagebox.showerror("Invalid choice", "Type must be 'first' or 'second'.")
            return
        try:
            new_sig = derivative(sig, kind=choice, name=unique_name(f"{sig.name}_{choice}Der", [s.name for s in self.signals]))
        except Exception as e:
            messagebox.showerror("Error", f"Derivative failed:\n{e}")
            return
        self.signals.append(new_sig)
        self._refresh_listbox()
        messagebox.showinfo("Done", f"{choice.capitalize()} derivative created: '{new_sig.name}'")

    # -----------------------
    # Shifting
    # -----------------------
    def shift_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal to shift.")
            return
        sig = sels[0]
        try:
            k_str = simpledialog.askstring("Shift", "Enter integer k (positive = delay, negative = advance):")
            if k_str is None:
                return
            k = int(k_str)
        except Exception:
            messagebox.showerror("Error", "Invalid integer k.")
            return
        try:
            new_sig = shift(sig, k, name=unique_name(f"{sig.name}_shift_{k}", [s.name for s in self.signals]))
        except Exception as e:
            messagebox.showerror("Error", f"Shift failed:\n{e}")
            return
        self.signals.append(new_sig)
        self._refresh_listbox()
        messagebox.showinfo("Done", f"Signal shifted by k={k}. New signal '{new_sig.name}' created.")

    # -----------------------
    # Folding
    # -----------------------
    def fold_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal to fold.")
            return
        sig = sels[0]
        try:
            new_sig = fold(sig, name=unique_name(f"{sig.name}_fold", [s.name for s in self.signals]))
        except Exception as e:
            messagebox.showerror("Error", f"Folding failed:\n{e}")
            return
        self.signals.append(new_sig)
        self._refresh_listbox()
        messagebox.showinfo("Done", f"Signal folded (time-reversed). New signal '{new_sig.name}' created.")

    # -----------------------
    # Fold then Shift
    # -----------------------
    def fold_shift_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal to fold and shift.")
            return
        sig = sels[0]
        try:
            k_str = simpledialog.askstring("Fold & Shift", "Enter integer k to shift folded signal (positive = delay):")
            if k_str is None:
                return
            k = int(k_str)
        except Exception:
            messagebox.showerror("Error", "Invalid integer k.")
            return
        try:
            new_sig = fold_and_shift(sig, k, name=unique_name(f"{sig.name}_fold_shift_{k}", [s.name for s in self.signals]))
        except Exception as e:
            messagebox.showerror("Error", f"Fold+Shift failed:\n{e}")
            return
        self.signals.append(new_sig)
        self._refresh_listbox()
        messagebox.showinfo("Done", f"Signal folded then shifted by k={k}. New signal '{new_sig.name}' created.")

    # -----------------------
    # Quantization
    # -----------------------
    def quantize_selected_signal(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal.")
            return
        sig = sels[0]
        try:
            bits = int(simpledialog.askstring("Quantization", "Enter number of bits:"))
        except:
            messagebox.showerror("Error", "Invalid bits.")
            return
        try:
            result = quantize_signal(sig, bits)
            result.name = unique_name(f"{sig.name}_Q{bits}", [s.name for s in self.signals])
            self.signals.append(result)
            self._refresh_listbox()
            messagebox.showinfo("Quantization", f"Created '{result.name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Quantization failed:\n{e}")

    # =======================
    # FREQUENCY DOMAIN
    # =======================
    def apply_dft(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal.")
            return

        sig = sels[0]
        try:
            y = [s[1] for s in sig.samples]
            fs = float(simpledialog.askstring("Sampling Frequency", "Enter Fs (Hz):"))
            if fs <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Invalid sampling frequency.")
            return

        try:
            (self.freq_components,
             self.frequencies,
             self.amplitudes_norm,
             self.amplitudes_unnorm,
             self.phases) = dft(y, fs)

            self.current_fs = fs
            self.selected_signal_for_ft = sig

            messagebox.showinfo("DFT", "DFT applied successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"DFT failed:\n{e}")

    def apply_idft(self):
        if self.freq_components is None:
            messagebox.showwarning("Warning", "Perform DFT first.")
            return

        try:
            y = idft(self.freq_components)
            x = [i for i in range(len(y))]
            result = Signal.from_arrays(
                x, y,
                name=unique_name("IDFT_Result", [s.name for s in self.signals])
            )
            self.signals.append(result)
            self._refresh_listbox()
            messagebox.showinfo("IDFT", f"Reconstructed signal '{result.name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"IDFT failed:\n{e}")

    def show_dominant_frequencies(self):
        if self.frequencies is None or self.amplitudes_norm is None:
            messagebox.showwarning("Warning", "Apply DFT first.")
            return

        try:
            dom = get_dominant_frequencies(self.frequencies, self.amplitudes_norm)
            if not dom:
                messagebox.showinfo("Dominant Frequencies", "No dominant frequencies found or DFT not available.")
                return
            messagebox.showinfo("Dominant Frequencies",
                                "\n".join([f"{f} Hz" for f in dom]))
        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{e}")

    def remove_dc(self):
        if self.amplitudes_norm is None:
            messagebox.showwarning("Warning", "Apply DFT first.")
            return

        try:
            self.amplitudes_norm[0] = 0
            self.amplitudes_unnorm[0] = 0
            messagebox.showinfo("DC Removal", "DC component removed.")
        except Exception as e:
            messagebox.showerror("Error", f"DC removal failed:\n{e}")

    def remove_dc_time_domain(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal.")
            return

        sig = sels[0]
        try:
            mean_val = sum([s[1] for s in sig.samples]) / len(sig.samples)
            new_samples = [[x, y - mean_val] for x, y in sig.samples]
            result = Signal(sig.signal_type, sig.is_periodic, new_samples,
                            name=unique_name(f"{sig.name}_noDC", [s.name for s in self.signals]))
            self.signals.append(result)
            self._refresh_listbox()
            messagebox.showinfo("Remove DC", f"Created '{result.name}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{e}")

    def modify_components_dialog(self):
        if self.freq_components is None:
            messagebox.showwarning("Warning", "Apply DFT first.")
            return

        try:
            idx = int(simpledialog.askstring("Modify Component",
                                             "Enter index of frequency component:"))
            amp = float(simpledialog.askstring("Amplitude", "Enter new amplitude:"))
            phase = float(simpledialog.askstring("Phase", "Enter new phase (radians):"))

            self.amplitudes_norm[idx] = amp
            self.phases[idx] = phase

            messagebox.showinfo("Modify", "Component modified.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{e}")

    def test_reconstruction_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal.")
            return

        sig = sels[0]
        file = filedialog.askopenfilename(
            title="Select reference file",
            filetypes=[("Text files", "*.txt")]
        )
        if not file:
            return

        try:
            test_reconstructed_signal(sig, file)
        except Exception as e:
            messagebox.showerror("Error", f"Test failed:\n{e}")

    def test_dft_output_dialog(self):
        if self.amplitudes_norm is None or self.phases is None:
            messagebox.showwarning("Warning", "Apply DFT first.")
            return

        file = filedialog.askopenfilename(
            title="Select test file",
            filetypes=[("Text files", "*.txt")]
        )
        if not file:
            return

        try:
            run_comparison_test(file, self.amplitudes_norm, self.phases)
        except Exception as e:
            messagebox.showerror("Error", f"Test failed:\n{e}")

    def filtering_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning('Warning', 'Select exactly one signal to filter.')
            return
        sig = sels[0]

        # ask filter type
        ftype = simpledialog.askstring(
            "Filter Type",
            "Enter filter type: 'low', 'high', 'bandpass', or 'bandstop'"
        )
        if ftype is None:
            return
        ftype = ftype.strip().lower()
        if ftype not in ('low','high','bandpass','bandstop'):
            messagebox.showerror('Invalid', 'Filter type must be low/high/bandpass/bandstop')
            return

        try:
            Fs_str = simpledialog.askstring('Sampling Frequency', 'Enter sampling frequency (Hz):')
            if Fs_str is None:
                return
            Fs = float(Fs_str)
            if Fs <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror('Error', 'Invalid sampling frequency')
            return

        try:
            if ftype in ('low','high'):
                fcut_str = simpledialog.askstring('Cutoff Frequency', 'Enter cutoff frequency (Hz):')
                if fcut_str is None:
                    return
                fcut = float(fcut_str)
                fcuts = fcut
            else:
                f1_str = simpledialog.askstring('Lower cutoff f1', 'Enter lower cutoff f1 (Hz):')
                if f1_str is None:
                    return
                f2_str = simpledialog.askstring('Upper cutoff f2', 'Enter upper cutoff f2 (Hz):')
                if f2_str is None:
                    return
                f1 = float(f1_str)
                f2 = float(f2_str)
                if f2 <= f1:
                    messagebox.showerror('Error', 'f2 must be > f1')
                    return
                fcuts = (f1, f2)

            A_s_str = simpledialog.askstring('Stopband attenuation', 'Enter desired stopband attenuation (dB), e.g. 60')
            if A_s_str is None:
                return
            A_s = float(A_s_str)

            trans_str = simpledialog.askstring('Transition band', 'Enter transition band width (Hz)')
            if trans_str is None:
                return
            trans = float(trans_str)
            if trans <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror('Error', 'Invalid numeric input')
            return

        # design
        try:
            h, meta = design_fir(ftype, Fs, fcuts, A_s, trans)
        except Exception as e:
            messagebox.showerror('Design error', f'Could not design filter:\\n{e}')
            return

        # Save coefficients
        try:
            savefile = save_coefficients(h)
        except Exception:
            savefile = None
        if savefile:
            messagebox.showinfo('Saved', f'Filter coefficients saved to:\\n{savefile}')

        # Apply filter (convolution)
        try:
            existing = [s.name for s in self.signals]
            base = f"{sig.name}_filtered"
            out_name = unique_name(base, existing)
            result = apply_filter_to_signal(sig, h, name=out_name)
            result.name = out_name
            self.signals.append(result)
            self._refresh_listbox()
            messagebox.showinfo('Done', f'Filter applied. New signal \"{result.name}\" created.')
        except Exception as e:
            messagebox.showerror('Filtering error', f'Filtering failed:\\n{e}')

    def resampling_dialog(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal.")
            return
    
        sig = sels[0]
    
        try:
            M = int(simpledialog.askstring("Resampling", "Enter M (decimation factor):"))
            L = int(simpledialog.askstring("Resampling", "Enter L (interpolation factor):"))
        except:
            messagebox.showerror("Error", "Invalid M or L value.")
            return
    
        try:
            from resampling import resample_signal
            new_sig = resample_signal(sig, M, L)
            new_sig.name = f"{sig.name}_R_{L}-{M}"
            self.signals.append(new_sig)
            self._refresh_listbox()
            messagebox.showinfo("Done", f"Resampling completed. New signal: {new_sig.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Resampling failed:\n{e}")
