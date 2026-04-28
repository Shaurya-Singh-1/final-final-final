from text_tools import slugify


def test_collapse_and_strip_dashes():
    assert slugify("  Hello,   World!!  ") == "hello-world"


def test_internal_words_remain_separated():
    assert slugify("A  B  C") == "a-b-c"
