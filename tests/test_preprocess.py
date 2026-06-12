from ccd.preprocess import normalize_hostname, is_valid_hostname


def test_normalize_hostname_basic():
    assert normalize_hostname("Example.COM") == "example.com"
    assert normalize_hostname("http://Example.COM:443/path") == "example.com"


def test_is_valid_hostname():
    assert is_valid_hostname("example.com")
    assert not is_valid_hostname("")
    assert not is_valid_hostname("a" * 300)
