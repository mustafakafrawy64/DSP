

import numpy as np
import math

# Import the user-provided comparison functions
# This assumes 'signalcompare.py' is in the same directory.
try:
    from signalcompare import SignalComapreAmplitude, SignalComaprePhaseShift
except ImportError:
    print("Warning: 'signalcompare.py' not found. Comparison test function will not work.")
    # Define dummy functions if not found, so the module can be imported
    def SignalComapreAmplitude(a, b): return False
    def SignalComaprePhaseShift(a, b): return False

# Import the comparison function from CompareSignals.py
try:
    from CompareSignals import SignalsAreEqual
except ImportError:
    print("Warning: 'CompareSignals.py' not found. Signal comparison test will not work.")
    def SignalsAreEqual(name, file, indices, samples):
        print(f"Error: Could not run test '{name}'. 'CompareSignals.py' not found.")



# ==========================================================
#      Manual Recursive FFT / IFFT (Decimation-in-Time)
# ==========================================================
def fft_manual(x):
    """Manual implementation of the Cooley–Tukey FFT (recursive)."""
    N = len(x)
    x = np.asarray(x, dtype=complex)

    if N <= 1:
        return x

    # Split even/odd
    even = fft_manual(x[::2])
    odd = fft_manual(x[1::2])

    # Twiddle factors
    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    return np.concatenate([
        even + factor[:N // 2] * odd,
        even - factor[:N // 2] * odd
    ])


def ifft_manual(X):
    """Manual implementation of the Inverse FFT (recursive)."""
    N = len(X)
    X = np.asarray(X, dtype=complex)
    if N <= 1:
        return X

    # Split even/odd
    even = ifft_manual(X[::2])
    odd = ifft_manual(X[1::2])

    # Twiddle factors (sign flipped)
    factor = np.exp(2j * np.pi * np.arange(N) / N)

    return np.concatenate([
        even + factor[:N // 2] * odd,
        even - factor[:N // 2] * odd
    ]) / 2


# ==========================================================
#                 Wrapper DFT/IDFT Interface
# ==========================================================
def dft(y_values, Fs):
    """
    Manual FFT wrapper returning frequency-domain data.
    Returns frequencies, normalized amplitude, phase, raw amplitude, complex FFT.
    Automatically saves a frequency-domain text file after computation.
    """
    N = len(y_values)
    if N == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    fft_complex = fft_manual(y_values)
    frequencies = np.arange(N) * Fs / N

    amplitudes_unnorm = np.abs(fft_complex)
    phases = np.angle(fft_complex)

    max_amp = np.max(amplitudes_unnorm)
    amplitudes_norm = amplitudes_unnorm / max_amp if max_amp != 0 else amplitudes_unnorm

    # ===============================================================
    # 🔽 AUTO-SAVE FREQUENCY DOMAIN RESULTS
    # ===============================================================
    try:
        output_filename = "fft_output_auto.txt"
        with open(output_filename, "w") as f:
            f.write("0\n")  # signal type placeholder
            f.write("0\n")  # periodic placeholder
            f.write(f"{N}\n")
            for i in range(N):
                f.write(f"{amplitudes_unnorm[i]:.6f} {phases[i]:.6f}\n")

        print(f"✅ FFT output automatically saved to: {output_filename}")

    except Exception as e:
        print(f"⚠️ Could not save FFT output file: {e}")

    return frequencies, amplitudes_norm, phases, amplitudes_unnorm, fft_complex



def idft(complex_components):
    """Manual Inverse FFT (real part only)."""
    result = ifft_manual(complex_components)
    return np.real(result)


def get_dominant_frequencies(frequencies, normalized_amplitudes, threshold=0.5):
    """
    Finds frequencies with a normalized amplitude above a given threshold.
    """
    N = len(frequencies)
    positive_freq_indices = np.where(normalized_amplitudes[:N//2] > threshold)[0]
    dominant_freqs = frequencies[positive_freq_indices]

    return list(dominant_freqs)

def run_comparison_test(output_file_to_read, input_amplitudes, input_phases):
    """
    Runs the comparison test as described in 'Signal compare function -- instructions.txt'.
    Uses 'signalcompare.py' for amplitude and phase.

    Returns:
        bool: True if both amplitude and phase tests pass, False otherwise.
    """

    expected_amplitudes = []
    expected_phases = []

    try:
        with open(output_file_to_read, 'r') as f:
            # Skip the 3-line header of the test file
            f.readline()
            f.readline()
            f.readline()

            line = f.readline()
            while line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        # Remove the 'f' character if it exists
                        amp_str = parts[0].replace('f', '')
                        phase_str = parts[1].replace('f', '')

                        expected_amplitudes.append(float(amp_str))
                        expected_phases.append(float(phase_str))
                    except ValueError as e:
                        print(f"Skipping malformed line in test file: {line.strip()} | Error: {e}")
                line = f.readline()

    except FileNotFoundError:
        print(f"Error: Test file not found at {output_file_to_read}")
        return False
    except Exception as e:
        print(f"Error reading test file: {e}")
        return False

    if not expected_amplitudes:
        print("Error: No valid data read from test file.")
        return False

    print("Running Signal Comparison Test (Amplitudes/Phases)...")

    # 3- run each function in file
    amp_test = SignalComapreAmplitude(input_amplitudes, expected_amplitudes)
    phase_test = SignalComaprePhaseShift(input_phases, expected_phases)

    # 4- and make condition if the two function return the true
    if amp_test and phase_test:
        print("Signal Compare Test PASSED")
        return True
    else:
        print(f"Signal Compare Test FAILED (Amplitude Pass: {amp_test}, Phase Pass: {phase_test})")
        return False


def test_reconstructed_signal(signal_object, test_file_path):
    """
    Tests a time-domain signal object against an expected signal file
    using the SignalsAreEqual function from CompareSignals.py.

    Prints results to the console.
    """
    if not signal_object or not signal_object.samples:
        print("Test Error: Invalid signal object provided.")
        return

    if not test_file_path:
        print("Test Error: No test file path provided.")
        return

    # Extract indices (X-values) and samples (Y-values)
    # Use int() on X-values to match the expected format for 'Your_indices'
    indices = [int(s[0]) for s in signal_object.samples]
    samples = [s[1] for s in signal_object.samples]

    print(f"\nRunning Time-Domain Signal Comparison Test...")
    print(f"  Signal: {signal_object.name}")
    print(f"  Test File: {test_file_path}")

    # Call the imported test function
    SignalsAreEqual(
        TaskName=f"'{signal_object.name}' Reconstruction Test",
        given_output_filePath=test_file_path,
        Your_indices=indices,
        Your_samples=samples
    )