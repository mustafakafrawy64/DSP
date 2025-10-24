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

# --- NEW IMPORT ---
# Import the new Fourier Transform functions
try:
    from Fourier_transform import dft, idft, get_dominant_frequencies
except ImportError:
    messagebox.showerror("Error", "Fourier_transform.py not found. Frequency domain features will be disabled.")


    # Define dummy functions to prevent crashes
    def dft(y, fs):
        return [], [], [], [], []


    def idft(c):
        return []


    def get_dominant_frequencies(f, a):
        return []


# --- END NEW IMPORT ---


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")
        self.root.geometry("1366x650")

        self.signals = []
        self.result_signal = None

        # --- NEW FT DATA STORAGE ---
        self.freq_components = None
        self.frequencies = None
        self.amplitudes_norm = None
        self.amplitudes_unnorm = None
        self.phases = None
        self.current_fs = None
        self.selected_signal_for_ft = None
        # --- END NEW FT DATA STORAGE ---

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
        menubar.add_cascade(label="Operations", menu=ops_menu)

        # ---Quantization menu---
        ops_menu.add_separator()
        ops_menu.add_command(label="Quantize Signal", command=self.quantize_selected_signal)

        # --- NEW FREQUENCY DOMAIN MENU ---
        ft_menu = tk.Menu(menubar, tearoff=0)
        ft_menu.add_command(label="Apply Fourier Transform (DFT)", command=self.apply_dft)
        ft_menu.add_command(label="Show Dominant Frequencies", command=self.show_dominant_frequencies)
        ft_menu.add_separator()
        ft_menu.add_command(label="Remove DC Component", command=self.remove_dc)
        ft_menu.add_command(label="Modify Components...", command=self.modify_components_dialog)
        ft_menu.add_separator()
        ft_menu.add_command(label="Reconstruct Signal (IDFT)", command=self.apply_idft)
        menubar.add_cascade(label="Frequency Domain", menu=ft_menu)
        # --- END NEW MENU ---

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
            from quantization import quantize_to_file

            if mode == "bits":
                bits = int(simpledialog.askstring("Bits", "Enter number of bits:"))
                test_file_1 = filedialog.askopenfilename(
                    title="Select Quan1_Out.txt (expected test file)",
                    filetypes=[("Text files", "*.txt")]
                )
                if not test_file_1:
                    messagebox.showinfo("Skipped", "No test file selected — skipping test.")
                    test_file_1 = None

                output_filename = filedialog.asksaveasfilename(
                    title="Save Quantization Output",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")],
                    initialfile="quantized_output.txt"
                )
                if not output_filename:
                    return

                _, codes, q_vals, _ = quantize_to_file(
                    sig.samples,
                    output_filename,
                    mode,
                    num_bits=bits,
                    test_file_1=test_file_1
                )
                messagebox.showinfo("Done", f"Quantization complete!\nSaved to:\n{output_filename}")

            else:
                levels = int(simpledialog.askstring("Levels", "Enter number of levels:"))
                test_file_2 = filedialog.askopenfilename(
                    title="Select Quan2_Out.txt (expected test file)",
                    filetypes=[("Text files", "*.txt")]
                )
                if not test_file_2:
                    messagebox.showinfo("Skipped", "No test file selected — skipping test.")
                    test_file_2 = None

                output_filename = filedialog.asksaveasfilename(
                    title="Save Quantization Output",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt")],
                    initialfile="quantized_output.txt"
                )
                if not output_filename:
                    return

                interval, codes, q_vals, errs = quantize_to_file(
                    sig.samples,
                    output_filename,
                    mode,
                    num_levels=levels,
                    test_file_2=test_file_2
                )
                messagebox.showinfo("Done", f"Quantization complete!\nSaved to:\n{output_filename}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =================================================
    # --- NEW FREQUENCY DOMAIN (TASK 3) FUNCTIONS ---
    # =================================================

    def _plot_frequency_domain(self):
        """Helper function to plot the current frequency domain data."""
        if self.frequencies is None:
            messagebox.showerror("Error", "No Frequency Domain data to plot.")
            return

        self.ax_cont.clear()
        self.ax_disc.clear()

        # We only plot the first half (positive frequencies)
        N = len(self.frequencies)
        n_to_show = N // 2

        # Plot Amplitude
        self.ax_cont.plot(self.frequencies[:n_to_show], self.amplitudes_norm[:n_to_show])
        self.ax_cont.set_title("Frequency vs. Amplitude (Normalized)")
        self.ax_cont.set_xlabel("Frequency (Hz)")
        self.ax_cont.set_ylabel("Normalized Amplitude")

        # Plot Phase
        self.ax_disc.stem(self.frequencies[:n_to_show], self.phases[:n_to_show])
        self.ax_disc.set_title("Frequency vs. Phase")
        self.ax_disc.set_xlabel("Frequency (Hz)")
        self.ax_disc.set_ylabel("Phase (Radians)")

        self.canvas.draw()

    def apply_dft(self):
        """
        Applies Fourier Transform to the selected signal and plots the result.
        """
        sels = self._get_selected_signals()
        if len(sels) != 1:
            messagebox.showwarning("Warning", "Select exactly one signal to transform.")
            return

        self.selected_signal_for_ft = sels[0]

        try:
            Fs_str = simpledialog.askstring("Sampling Frequency", "Enter Sampling Frequency (Hz):")
            if Fs_str is None: return  # User cancelled
            Fs = float(Fs_str)
            if Fs <= 0:
                raise ValueError("Sampling frequency must be positive.")
        except (TypeError, ValueError) as e:
            messagebox.showerror("Error", f"Invalid input for Sampling Frequency: {e}")
            return

        self.current_fs = Fs
        y_values = [s[1] for s in self.selected_signal_for_ft.samples]

        # Call the DFT function
        (self.frequencies,
         self.amplitudes_norm,
         self.phases,
         self.amplitudes_unnorm,
         self.freq_components) = dft(y_values, self.current_fs)

        if len(self.frequencies) == 0:
            messagebox.showerror("Error", "DFT computation failed. Signal may be empty.")
            return

        # Plot the results
        self._plot_frequency_domain()

        # Also show dominant frequencies
        self.show_dominant_frequencies(show_message=True)

    def show_dominant_frequencies(self, show_message=False):
        """
        Displays the dominant frequencies (Amp > 0.5) in a messagebox.
        """
        if self.frequencies is None:
            messagebox.showwarning("No FT Data", "Please apply Fourier Transform first.")
            return

        dominant_freqs = get_dominant_frequencies(self.frequencies, self.amplitudes_norm, threshold=0.5)

        if show_message:  # To avoid double popups when called from apply_dft
            return

        if not dominant_freqs:
            messagebox.showinfo("Dominant Frequencies", "No dominant frequencies found (Amplitude > 0.5).")
        else:
            freq_str = "\n".join([f"{f:.2f} Hz" for f in dominant_freqs])
            messagebox.showinfo("Dominant Frequencies", f"Frequencies with Amplitude > 0.5:\n{freq_str}")

    def remove_dc(self):
        """
        Removes the DC component (F(0)) from the stored FT data and replots.
        """
        if self.freq_components is None:
            messagebox.showwarning("No FT Data", "Please apply Fourier Transform first.")
            return

        # Set the 0-frequency component to 0
        self.freq_components[0] = 0

        # Recalculate amplitudes and phases
        self.amplitudes_unnorm = np.abs(self.freq_components)
        self.phases = np.angle(self.freq_components)

        max_amp = np.max(self.amplitudes_unnorm)
        if max_amp == 0:
            self.amplitudes_norm = self.amplitudes_unnorm
        else:
            self.amplitudes_norm = self.amplitudes_unnorm / max_amp

        # Re-plot
        self._plot_frequency_domain()
        messagebox.showinfo("DC Removed", "DC component (F(0)) has been set to zero.\nRe-plotting frequency domain.")

    def modify_components_dialog(self):
        """
        Opens a new window to modify the amplitude and phase of FT components.
        """
        if self.frequencies is None:
            messagebox.showwarning("No FT Data", "Please apply Fourier Transform first.")
            return

        # --- Create Toplevel Window ---
        top = tk.Toplevel(self.root)
        top.title("Modify Frequency Components")
        top.transient(self.root)  # Keep it on top
        top.grab_set()  # Modal

        # --- Data Store (as a list of tuples) ---
        # We work on a temporary copy
        current_data = list(zip(
            self.frequencies,
            self.amplitudes_unnorm,
            self.phases
        ))

        # --- Listbox ---
        list_frame = tk.Frame(top)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(list_frame, text="Components (Index | Freq | Amp | Phase):").pack(anchor='w')

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(list_frame, width=55, height=20, yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.pack(fill=tk.BOTH, expand=True)

        def populate_listbox():
            listbox.delete(0, tk.END)
            # Show all N components
            for i, (f, a, p) in enumerate(current_data):
                listbox.insert(tk.END, f"{i:03d} | F: {f:8.2f} Hz | A: {a:8.2f} | P: {p:8.2f} rad")

        # --- Edit Frame ---
        edit_frame = tk.Frame(top)
        edit_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        tk.Label(edit_frame, text="New Amplitude:").pack()
        amp_entry = tk.Entry(edit_frame)
        amp_entry.pack()

        tk.Label(edit_frame, text="New Phase (rad):").pack()
        phase_entry = tk.Entry(edit_frame)
        phase_entry.pack()

        def on_select(evt):
            try:
                selected_index = listbox.curselection()[0]
                _, amp, phase = current_data[selected_index]
                amp_entry.delete(0, tk.END)
                amp_entry.insert(0, f"{amp:.4f}")
                phase_entry.delete(0, tk.END)
                phase_entry.insert(0, f"{phase:.4f}")
            except IndexError:
                pass  # No selection

        listbox.bind('<<ListboxSelect>>', on_select)

        def apply_modification():
            try:
                selected_index = listbox.curselection()[0]
                new_amp = float(amp_entry.get())
                new_phase = float(phase_entry.get())

                # Update temporary data store
                freq, _, _ = current_data[selected_index]
                current_data[selected_index] = (freq, new_amp, new_phase)

                # Update listbox
                populate_listbox()
                listbox.selection_set(selected_index)
            except Exception as e:
                messagebox.showerror("Error", f"Could not apply modification: {e}", parent=top)

        tk.Button(edit_frame, text="Apply Modification", command=apply_modification).pack(pady=10)

        # --- Done/Cancel Buttons ---
        def on_done():
            # Write changes back to self
            self.frequencies = np.array([d[0] for d in current_data])
            self.amplitudes_unnorm = np.array([d[1] for d in current_data])
            self.phases = np.array([d[2] for d in current_data])

            # Recalculate normalized amps
            max_amp = np.max(self.amplitudes_unnorm)
            self.amplitudes_norm = self.amplitudes_unnorm / max_amp if max_amp != 0 else self.amplitudes_unnorm

            # Re-plot FT
            self._plot_frequency_domain()
            messagebox.showinfo("Success", "Components have been updated.", parent=self.root)
            top.destroy()

        def on_cancel():
            top.destroy()

        tk.Button(edit_frame, text="Done (Save Changes)", command=on_done).pack(side=tk.BOTTOM, pady=20)
        tk.Button(edit_frame, text="Cancel", command=on_cancel).pack(side=tk.BOTTOM)

        # --- Initial Population ---
        populate_listbox()

    def apply_idft(self):
        """
        Reconstructs the signal from the (potentially modified) FT data
        and adds it as a new signal to the list.
        """
        if self.amplitudes_unnorm is None:
            messagebox.showwarning("No FT Data", "Please apply Fourier Transform or modify components first.")
            return

        if self.selected_signal_for_ft is None:
            messagebox.showerror("Error", "Original signal context lost. Please re-apply DFT.")
            return

        # Reconstruct the complex array from the (potentially modified) amps/phases
        # This is the crucial step
        modified_complex = self.amplitudes_unnorm * np.exp(1j * self.phases)

        # Call IDFT
        reconstructed_y = idft(modified_complex)

        # Get the original x_values (time/index)
        original_samples = self.selected_signal_for_ft.samples
        if len(original_samples) != len(reconstructed_y):
            messagebox.showerror("Error", "Mismatch in length during reconstruction. This should not happen.")
            return

        x_values = [s[0] for s in original_samples]

        # Create a new Signal object
        sig_name = unique_name(
            f"{self.selected_signal_for_ft.name}_IDFT",
            [s.name for s in self.signals]
        )
        new_sig = Signal.from_arrays(x_values, reconstructed_y, name=sig_name)

        self.signals.append(new_sig)
        self._refresh_listbox()

        messagebox.showinfo("Signal Reconstructed",
                            f"Signal '{sig_name}' created from IDFT.\nSelect it from the list and press 'Plot Selected' to view.")