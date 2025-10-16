import numpy as np

def GenerateSignal(signal_type, A, AnalogFrequency, SamplingFrequency, PhaseShift, duration=1.0):
    Ts = 1.0 / SamplingFrequency
    n = np.arange(0.0, duration, Ts)
    if signal_type == "sin":
        x = A * np.sin(2 * np.pi * AnalogFrequency * n + PhaseShift)
    else:
        x = A * np.cos(2 * np.pi * AnalogFrequency * n + PhaseShift)
    return n, x
