"""Checks for the generated manuscript comparison tables."""

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
        "### Table 4.",
    ):
        assert title in readme


def test_sparkseg_kits23_metrics_are_best():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert '<strong><span style="color:#d62728">85.66</span></strong>' in readme
    assert '<strong><span style="color:#d62728">13.41</span></strong>' in readme
