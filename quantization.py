import numpy as np

# ==========================================================
#                Quantization Test Functions (as given)
# ==========================================================
def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one") 
            return
    print("QuantizationTest1 Test case passed successfully")



def QuantizationTest2(file_name,Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError):
    expectedIntervalIndices=[]
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    expectedSampledError=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==4:
                L=line.split(' ')
                V1=int(L[0])
                V2=str(L[1])
                V3=float(L[2])
                V4=float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
     or len(Your_EncodedValues)!=len(expectedEncodedValues)
      or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
      or len(Your_SampledError)!=len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
            return
        
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one") 
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one") 
            return
    print("QuantizationTest2 Test case passed successfully")


# ==========================================================
#                Quantization Core Function
# ==========================================================
def quantize_signal(samples, num_levels=None, num_bits=None):
    y = np.array([s[1] for s in samples])

    if num_bits is not None:
        num_levels = 2 ** num_bits
    elif num_levels is None:
        raise ValueError("Must provide either num_levels or num_bits")

    y_min, y_max = np.min(y), np.max(y)
    delta = (y_max - y_min) / num_levels

    # Mid-rise quantization levels
    levels = y_min + delta * (np.arange(num_levels) + 0.5)

    interval_indices, encoded_values, quantized_values, errors = [], [], [], []

    for sample in y:
        idx = int((sample - y_min) / delta)
        if idx >= num_levels:
            idx = num_levels - 1
        q = levels[idx]
        err = sample - q
        code = format(idx, f"0{int(np.ceil(np.log2(num_levels)))}b")

        interval_indices.append(idx)
        encoded_values.append(code)
        quantized_values.append(round(q, 3))
        errors.append(round(err, 3))

    return interval_indices, encoded_values, quantized_values, errors


# ==========================================================
#                Quantize and Write to File + Run Tests
# ==========================================================
def quantize_to_file(samples, output_filename, mode,
                     num_bits=None, num_levels=None,
                     test_file_1=None, test_file_2=None):
    """
    Quantizes samples, writes output file, and automatically
    runs QuantizationTest1 or QuantizationTest2 (showing results in terminal).
    """

    interval_indices, encoded_values, quantized_values, errors = quantize_signal(
        samples, num_bits=num_bits, num_levels=num_levels
    )

    with open(output_filename, 'w') as f:
        f.write("0\n")
        f.write("0\n")
        f.write(f"{len(samples)}\n")

        if mode == "bits":
            for code, q in zip(encoded_values, quantized_values):
                f.write(f"{code} {q}\n")
        elif mode == "levels":
            for idx, code, q, err in zip(interval_indices, encoded_values, quantized_values, errors):
                f.write(f"{idx} {code} {q} {err}\n")

    print(f"\nQuantized data saved to: {output_filename}")

    # ============================
    # Automatically run test cases
    # ============================
    if mode == "bits" and test_file_1:
        print("\nRunning QuantizationTest1...")
        QuantizationTest1(test_file_1, encoded_values, quantized_values)

    elif mode == "levels" and test_file_2:
        print("\nRunning QuantizationTest2...")
        QuantizationTest2(test_file_2, interval_indices, encoded_values, quantized_values, errors)

    return interval_indices, encoded_values, quantized_values, errors
