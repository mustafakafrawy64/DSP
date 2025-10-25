import math

# Use to test the Amplitude of DFT and IDFT
def SignalComapreAmplitude(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            # --- MODIFIED LOGIC ---
            A = round(SignalInput[i], 13)
            B = round(SignalOutput[i], 13)
            
            if A != B:
                print(f"Amplitude Mismatch at index {i}: Got {A}, expected {B}")
                return False
        return True

def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))

# Use to test the PhaseShift of DFT
def SignalComaprePhaseShift(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = round(SignalInput[i], 9)
            B = round(SignalOutput[i], 9)

            if A != B:
                print(f"Phase Mismatch at index {i}: Got {A}, expected {B}")
                return False
        return True