from versioning import version_sort_key


def test_numeric_segments_sort_correctly():
    ordered = sorted(["1.2.0", "1.10.0", "1.3.0"], key=version_sort_key)
    assert ordered == ["1.2.0", "1.3.0", "1.10.0"]
