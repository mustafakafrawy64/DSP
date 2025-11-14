import numpy as np
from tkinter import simpledialog, filedialog, messagebox
from signal_model import Signal
import matplotlib.pyplot as plt
from Fourier_transform import dft, idft, run_comparison_test, test_reconstructed_signal


def fft_ifft_handler(signals):
    """
    Handles both:
      1️⃣ FFT (Decimation-in-Time) → show Frequency vs Amplitude/Phase.
      2️⃣ IFFT reconstruction from a polar frequency file.
    """

    choice = simpledialog.askstring(
        "FFT / IFFT / Test",
        "Enter 'fft' to compute FFT of a signal\n"
        "Enter 'ifft' to reconstruct from a frequency file\n"
    )
    if not choice:
        return

    # ===============================================================
    # 1️⃣ FFT MODE (Decimation-in-Time)
    # ===============================================================
    if choice.lower() == "fft":
        if len(signals) != 1:
            messagebox.showwarning("Warning", "Please select ONE signal for FFT.")
            return

        sig = signals[0]
        Fs = simpledialog.askfloat("Sampling Frequency", "Enter sampling frequency (Hz):")
        if not Fs or Fs <= 0:
            messagebox.showerror("Error", "Invalid sampling frequency.")
            return

        y = np.array([s[1] for s in sig.samples])
        freqs, amp_norm, phase, amp_raw, fft_complex = dft(y, Fs)

               # --- Close any previous plots before showing new FFT ---
        plt.close('all')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(freqs, amp_raw)
        ax1.set_title("Frequency vs Amplitude (Manual FFT)")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        ax2.plot(freqs, phase)
        ax2.set_title("Frequency vs Phase (Manual FFT)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (radians)")
        ax2.grid(True)

        fig.tight_layout()
        plt.show()


    # ===============================================================
    # 2️⃣ IFFT MODE (Reconstruction)
    # ===============================================================
    elif choice.lower() == "ifft":
        file_path = filedialog.askopenfilename(
            title="Select frequency-domain file (polar form)",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            lines = lines[3:]  # skip headers

            amplitudes, phases = [], []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    amplitudes.append(float(parts[0]))
                    phases.append(float(parts[1]))

            complex_components = np.array(amplitudes) * np.exp(1j * np.array(phases))
            reconstructed_y = idft(complex_components)

            x_vals = np.arange(len(reconstructed_y))
            reconstructed_signal = Signal.from_arrays(x_vals, reconstructed_y, name="Reconstructed_Signal")

            save_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt")],
                initialfile="reconstructed_signal.txt"
            )
            if save_path:
                reconstructed_signal.save_to_file(save_path)

                  
            plt.close('all')

            plt.figure(figsize=(8, 5))
            plt.plot(x_vals, reconstructed_y, label="Reconstructed Signal")
            plt.title("Time-Domain Reconstructed Signal (Manual IFFT)")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


        except Exception as e:
            messagebox.showerror("Error", f"Failed to reconstruct signal:\n{e}")

    
