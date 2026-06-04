from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_uses_current_deidentification_cli_flags():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "--public-release deidentification_release/data/release/hib_release.jsonl" in readme
    assert "--audit-dir deidentification_release/data/audits" in readme
    assert "--release deidentification_release/data/release/hib_release.jsonl" not in readme


def test_root_readme_certification_example_includes_radius():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("ccd certify \\")
    end = readme.index("```", start)
    snippet = readme[start:end]

    assert "--radius 1" in snippet
