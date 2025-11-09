import numpy as np
from tkinter import simpledialog, filedialog, messagebox
from signal_model import Signal
import matplotlib.pyplot as plt
from Fourier_transform import dft, idft  # dft now uses your manual recursive DIT FFT


def fft_ifft_handler(signals):
    """
    Handles both:
      1️⃣ FFT (Decimation-in-Time) → show Frequency vs Amplitude/Phase.
      2️⃣ IFFT reconstruction from a polar frequency file.
      3️⃣ Test mode → internal verification of both FFT and IFFT logic.
    """

    # Ask user for operation type
    choice = simpledialog.askstring(
        "FFT / IFFT / Test",
        "Enter 'fft' to compute FFT of a signal\n"
        "Enter 'ifft' to reconstruct from a frequency file\n"
        "Enter 'test' to run a built-in self-test:"
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

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(freqs, amp_raw)
        plt.title("Frequency vs Amplitude (Manual DIT FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(freqs, phase)
        plt.title("Frequency vs Phase (Manual DIT FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.grid(True)

        plt.tight_layout()
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

            plt.figure()
            plt.plot(x_vals, reconstructed_y, label="Reconstructed Signal")
            plt.title("Time-Domain Reconstructed Signal (IFFT)")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to reconstruct signal:\n{e}")

    # ===============================================================
    # 3️⃣ TEST MODE (built-in validation)
    # ===============================================================
    elif choice.lower() == "test":
        try:
            print("\n===== TESTING FFT/IFFT MODULE =====")

            # ---- Create a known sine wave ----
            Fs = 32
            t = np.arange(32)
            f0 = 4  # 4 Hz component
            y = np.sin(2 * np.pi * f0 * t / Fs)
            print(f"Generated sine wave: f = {f0} Hz, Fs = {Fs} Hz")

            # ---- Apply your manual FFT ----
            freqs, amp_norm, phase, amp_raw, fft_complex = dft(y, Fs)

            # Find the frequency with maximum amplitude
            dominant_freq = abs(freqs[np.argmax(amp_raw)])
            print(f"Detected dominant frequency: {dominant_freq:.2f} Hz")

            # ---- IFFT reconstruction ----
            reconstructed = idft(fft_complex)
            err = np.mean(np.abs(reconstructed - y))
            print(f"Reconstruction mean absolute error: {err:.6f}")

            # ---- Pass/Fail messages ----
            if abs(dominant_freq - f0) < 0.1 and err < 1e-6:
                messagebox.showinfo("FFT/IFFT Test", "✅ Test Passed!\nFFT and IFFT working correctly.")
            else:
                messagebox.showwarning("FFT/IFFT Test", "⚠️ Test completed but results deviate from expected.")

            # Plot test results
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(freqs, amp_raw)
            plt.title("Test FFT Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(t, y, label="Original")
            plt.plot(t, reconstructed, '--', label="Reconstructed")
            plt.title("Test: Original vs IFFT-Reconstructed")
            plt.xlabel("Sample index")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Test Error", f"Test failed:\n{e}")

    # ===============================================================
    else:
        messagebox.showerror("Error", "Invalid choice. Please enter 'fft', 'ifft', or 'test'.")
