"""Checks for the paper tables embedded in the README."""

from __future__ import annotations

from pathlib import Path


def test_paper_tables_cover_all_comparisons():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    for title in (
        "### Table 1.",
        "### Table 2.",
        "### Table 3a.",
        "### Table 3b.",
        "### Table 3c.",
        "### Table 4.",
    ):
        assert title in readme


def test_comparison_tables_are_editable_markdown():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "assets/paper/table" not in readme
    assert readme.count("| :--- |") >= 8
    for value in ("91.98", "91.73", "85.94", "85.66", "13.41"):
        assert value in readme
    for row in ("| nnU-Net [1] |", "| **SparkSeg** |", "| nnU-Net ResEnc [22] |"):
        assert row in readme
