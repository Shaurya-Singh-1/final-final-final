import pytest

from boolparse import parse_bool


def test_accepts_whitespace_and_case():
    assert parse_bool("  TRUE  ") is True
    assert parse_bool("\nFalse\t") is False


def test_invalid_value_raises():
    with pytest.raises(ValueError):
        parse_bool("definitely")
