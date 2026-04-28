def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(values):
        return []

    result = []
    for idx in range(0, len(values) - window):
        chunk = values[idx : idx + window]
        result.append(sum(chunk) / window)
    return result
