import json
from pathlib import Path

from ccd.corpus import (
    filter_hostnames,
    read_hostnames_from_benign_dir,
    read_hostnames_from_jsonl_dir,
    read_hostnames_from_txt_dir,
)


def test_filter_hostnames_dedup_and_length():
    hosts = ["a.com", "longhost.com", "otherlong.com", "longhost.com", "short"]
    out = filter_hostnames(hosts, min_length=5, dedup=True)
    assert out == ["longhost.com", "otherlong.com"]
    assert "short" not in out


def test_read_hostnames_from_jsonl_dir(tmp_path: Path):
    data = [{"hostname": "bad.com"}, {"hostname": "evil.com"}, {"other": "x"}]
    jsonl = tmp_path / "data.jsonl"
    with jsonl.open("w") as handle:
        for row in data:
            handle.write(json.dumps(row) + "\n")
    out = read_hostnames_from_jsonl_dir(tmp_path, key="hostname")
    assert "bad.com" in out
    assert "evil.com" in out
    assert len(out) == 2


def test_read_hostnames_from_txt_dir_with_csv(tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_text("one.com\ntwo.com\n")
    csv_path = tmp_path / "b.csv"
    csv_path.write_text("Hostname\nthree.com\n")
    out = read_hostnames_from_txt_dir(tmp_path, include_csv=True, csv_column="Hostname")
    assert set(out) == {"one.com", "two.com", "three.com"}


def test_read_hostnames_from_benign_dir(tmp_path: Path):
    benign_dir = tmp_path / "benign"
    benign_dir.mkdir()
    (benign_dir / "a.txt").write_text("good.com\n")
    (benign_dir / "b.txt").write_text("ok.com\n")
    nested = benign_dir / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("nested.com\n")
    out = read_hostnames_from_benign_dir(benign_dir)
    assert out == ["good.com", "ok.com", "nested.com"]
