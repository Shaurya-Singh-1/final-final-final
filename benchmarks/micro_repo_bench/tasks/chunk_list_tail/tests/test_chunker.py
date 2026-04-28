from chunker import chunk_list


def test_keeps_partial_tail_chunk():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_exact_division_still_works():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_invalid_size_raises():
    try:
        chunk_list([1, 2], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
