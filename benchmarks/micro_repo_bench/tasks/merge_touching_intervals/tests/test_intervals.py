from intervals import merge_intervals


def test_merges_touching_intervals():
    assert merge_intervals([(1, 3), (3, 5), (10, 12)]) == [(1, 5), (10, 12)]


def test_keeps_separate_intervals():
    assert merge_intervals([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]
