"""Checks for the rendered comparison tables."""

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


def test_comparison_tables_are_rendered_as_assets():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    for filename in (
        "table1_brats2021_v5.png",
        "table2_acdc_v5.png",
        "table3a_synapse_dsc_v5.png",
        "table3b_synapse_hd95_resources_v5.png",
        "table4_kits23_v5.png",
    ):
        relative = f"assets/paper/{filename}"
        assert relative in readme
        assert (Path(__file__).resolve().parents[1] / relative).is_file()
