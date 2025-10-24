import numpy as np
import math

# Import the user-provided comparison functions
try:
    from signalcompare import SignalComapreAmplitude, SignalComaprePhaseShift
except ImportError:
    print("Warning: 'signalcompare.py' not found. Comparison test function will not work.")


    # Define dummy functions if not found, so the module can be imported
    def SignalComapreAmplitude(a, b):
        return False


    def SignalComaprePhaseShift(a, b):
        return False

# Import the comparison function from CompareSignals.py
try:
    from CompareSignals import SignalsAreEqual
except ImportError:
    print("Warning: 'CompareSignals.py' not found. Signal comparison test will not work.")


    def SignalsAreEqual(name, file, indices, samples):
        print(f"Error: Could not run test '{name}'. 'CompareSignals.py' not found.")


# --- END NEW IMPORT ---


def dft(y_values, Fs):
    """
    Computes the Discrete Fourier Transform (DFT) of a signal.

    Returns:
        tuple: 
            - frequencies (np.array): Frequency bins.
            - amplitudes_norm (np.array): Amplitudes normalized from 0 to 1.
            - phases (np.array): Phases in radians.
            - amplitudes_unnorm (np.array): Unnormalized amplitudes.
            - fft_complex (np.array): The raw complex result from np.fft.fft.
    """
    N = len(y_values)
    if N == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Compute the FFT
    fft_complex = np.fft.fft(y_values)

    # Compute the corresponding frequencies
    # np.fft.fftfreq generates frequencies from [0, +Fs/2, -Fs/2, ..., -f]
    frequencies = np.fft.fftfreq(N, 1.0 / Fs)

    # Compute unnormalized amplitudes and phases
    amplitudes_unnorm = np.abs(fft_complex)
    phases = np.angle(fft_complex)

    # Normalize amplitudes
    max_amp = np.max(amplitudes_unnorm)
    if max_amp == 0:
        amplitudes_norm = amplitudes_unnorm
    else:
        amplitudes_norm = amplitudes_unnorm / max_amp

    # Return all components
    # We only plot the first N/2 components, but return all for reconstruction
    return frequencies, amplitudes_norm, phases, amplitudes_unnorm, fft_complex


def idft(complex_components):
    """
    Computes the Inverse Discrete Fourier Transform (IDFT).

    Returns:
        np.array: The reconstructed time-domain signal (real part).
    """
    ifft_result = np.fft.ifft(complex_components)

    # Return the real part, as the input signal was real
    return np.real(ifft_result)


def get_dominant_frequencies(frequencies, normalized_amplitudes, threshold=0.5):
    """
    Finds frequencies with a normalized amplitude above a given threshold.

    Returns:
        list: A list of dominant frequencies.
    """
    # Find indices where amplitude is above the threshold
    # We only check the positive frequencies (first half of the array)
    N = len(frequencies)
    positive_freq_indices = np.where(normalized_amplitudes[:N // 2] > threshold)[0]
    dominant_freqs = frequencies[positive_freq_indices]

    return list(dominant_freqs)


def run_comparison_test(output_file_to_read, input_amplitudes, input_phases):
    """
    Runs the comparison test as described in 'Signal compare function -- instructions.txt'.
    Uses 'signalcompare.py'.

    Returns:
        bool: True if both amplitude and phase tests pass, False otherwise.
    """


    expected_amplitudes = []
    expected_phases = []

    try:
        with open(output_file_to_read, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        expected_amplitudes.append(float(parts[0]))
                        expected_phases.append(float(parts[1]))
                    except ValueError:
                        print(f"Skipping malformed line in test file: {line}")

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
        print("Note: The provided 'signalcompare.py' may have bugs (e.g., in rounding phases).")
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

    # Extract indices (0, 1, 2, ...) and samples (y-values)
    # This matches the expected input for SignalsAreEqual
    indices = list(range(len(signal_object.samples)))
    samples = [s[1] for s in signal_object.samples]

    print(f"\nRunning Time-Domain Signal Comparison Test...")
    print(f"  Signal: {signal_object.name}")
    print(f"  Test File: {test_file_path}")

    # Call the imported test function
    # It will print "passed" or "failed" to the console
    SignalsAreEqual(
        TaskName=f"'{signal_object.name}' Reconstruction Test",
        given_output_filePath=test_file_path,
        Your_indices=indices,
        Your_samples=samples
    )