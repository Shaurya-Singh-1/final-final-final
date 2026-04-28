def chunk_list(items, size):
    if size <= 0:
        raise ValueError("size must be positive")

    chunks = []
    for idx in range(0, len(items) - size, size):
        chunks.append(items[idx : idx + size])
    return chunks
