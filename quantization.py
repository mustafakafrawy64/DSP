import numpy as np

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
#                Quantize and Write to File
# ==========================================================
def quantize_to_file(samples, output_filename, mode, num_bits=None, num_levels=None):
    """Quantizes given samples and writes results to a file with the correct format."""

    interval_indices, encoded_values, quantized_values, errors = quantize_signal(
        samples, num_bits=num_bits, num_levels=num_levels
    )

    with open(output_filename, 'w') as f:
        f.write("0\n")           # signal type placeholder
        f.write("0\n")           # periodic placeholder
        f.write(f"{len(samples)}\n")

        if mode == "bits":
            for code, q in zip(encoded_values, quantized_values):
                f.write(f"{code} {q}\n")
        elif mode == "levels":
            for idx, code, q, err in zip(interval_indices, encoded_values, quantized_values, errors):
                f.write(f"{idx} {code} {q} {err}\n")

    return interval_indices, encoded_values, quantized_values, errors
