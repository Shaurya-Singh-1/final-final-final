from averages import moving_average


def test_includes_last_window():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_large_window_returns_empty():
    assert moving_average([1, 2], 3) == []
