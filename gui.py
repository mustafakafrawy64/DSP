import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from signal_model import Signal
from signal_generator import GenerateSignal
from utils import unique_name
from quantization import quantize_signal
from signal_model import Signal


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")
        self.root.geometry("1366x650")

        self.signals = []
        self.result_signal = None

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
        view_menu.add_command(label="Plot Selected", command=self.plot_selected)
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
        menubar.add_cascade(label="Operations", menu=ops_menu)

        #---Quantization menu---
        ops_menu.add_separator()
        ops_menu.add_command(label="Quantize Signal", command=self.quantize_selected_signal)

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

    def save_selected(self):
        selected = self._get_selected_signals()
        if not selected:
            return
        for sig in selected:
            file = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt")],
                                                initialfile=f"{sig.name}.txt")
            if file:
                sig.save_to_file(file)

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

        self.ax_cont.set_title("Continuous")
        self.ax_disc.set_title("Discrete")
        self.ax_disc.set_xlabel("Time")
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
            c = float(simpledialog.askstring("Multiply by", "Enter constant:"))
        except (TypeError, ValueError):
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

    
        
    # =======================
    # QUANTIZATION
    # =======================
    def quantize_selected_signal(self):
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select one signal to quantize.")
            return

        sig = sels[0]

        mode = simpledialog.askstring("Quantization Mode", "Enter 'bits' or 'levels':")
        if mode not in ("bits", "levels"):
            messagebox.showerror("Invalid Input", "You must type 'bits' or 'levels'.")
            return

        try:
            if mode == "bits":
                bits = int(simpledialog.askstring("Bits", "Enter number of bits:"))
                output_filename = filedialog.asksaveasfilename(
                    title="Save Quantization Output",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")],
                    initialfile="quantized_output.txt"
                )
                if not output_filename:
                    return
                from quantization import quantize_to_file
                _, codes, q_vals, _ = quantize_to_file(sig.samples, output_filename, mode, num_bits=bits)
                messagebox.showinfo("Done", f"Quantization complete!\nSaved to:\n{output_filename}")

            else:  # levels mode
                levels = int(simpledialog.askstring("Levels", "Enter number of levels:"))
                output_filename = filedialog.asksaveasfilename(
                    title="Save Quantization Output",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")],
                    initialfile="quantized_output.txt"
                )
                if not output_filename:
                    return
                from quantization import quantize_to_file
                interval, codes, q_vals, errs = quantize_to_file(sig.samples, output_filename, mode, num_levels=levels)
                messagebox.showinfo("Done", f"Quantization complete!\nSaved to:\n{output_filename}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
