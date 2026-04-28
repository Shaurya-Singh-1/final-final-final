def median(values):
    ordered = sorted(values)
    size = len(ordered)
    if size == 0:
        raise ValueError("median() arg is an empty sequence")

    mid = size // 2
    if size % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1])
