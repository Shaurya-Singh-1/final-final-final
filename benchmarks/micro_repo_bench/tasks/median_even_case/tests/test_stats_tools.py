from stats_tools import median


def test_even_length_uses_average():
    assert median([10, 2, 8, 4]) == 6.0


def test_odd_length_still_works():
    assert median([3, 1, 2]) == 2.0
