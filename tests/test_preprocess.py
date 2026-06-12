from ccd.preprocess import is_valid_hostname, normalize_hostname, normalization_trace


def test_normalize_hostname_basic():
    assert normalize_hostname("Example.COM") == "example.com"
    assert normalize_hostname("http://Example.COM:443/path") == "example.com"
    assert normalize_hostname("Example.COM/path?x=1#frag") == "example.com"


def test_normalization_trace_records_url_segments_and_decoding():
    trace = normalization_trace("https://user:pw@caf%C3%A9.Example:443/path?x=1#frag")

    assert trace["normalized_hostname"] == "café.example"
    assert trace["percent_decode_changed"] is True
    assert trace["host_before_percent_decode"] == "caf%C3%A9.Example"
    assert trace["host_after_percent_decode"] == "café.Example"
    assert trace["segmentation"] == {
        "scheme": "https",
        "authority_present": True,
        "userinfo_present": True,
        "port_present": True,
        "path_present": True,
        "query_present": True,
        "fragment_present": True,
        "bracketed_ipv6": False,
    }


def test_normalize_hostname_decodes_utf8_percent_runs():
    assert normalize_hostname("caf%C3%A9.example") == "café.example"


def test_normalize_hostname_preserves_mixed_decode_residue_fallback():
    assert normalize_hostname("%FF.example", idna_roundtrip=False) == "ÿ.example"


def test_is_valid_hostname():
    assert is_valid_hostname("example.com")
    assert not is_valid_hostname("")
    assert not is_valid_hostname("a" * 300)
