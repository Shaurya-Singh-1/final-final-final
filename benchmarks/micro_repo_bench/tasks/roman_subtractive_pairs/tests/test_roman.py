from roman import roman_to_int


def test_subtractive_pairs():
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("XL") == 40
    assert roman_to_int("CM") == 900


def test_regular_numerals_still_work():
    assert roman_to_int("VIII") == 8
