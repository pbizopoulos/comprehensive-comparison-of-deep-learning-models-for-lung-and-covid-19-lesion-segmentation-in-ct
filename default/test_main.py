# Copyright (c) 2026- Paschalis Bizopoulos
"""Specify the research pipeline's offline example workflow."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_smoke_run_generates_readable_results_and_a_complete_manuscript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pytest workflow produces CSV results and a compiled PDF offline."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from default import main as subject  # noqa: PLC0415

    output = tmp_path / (
        "github.com/pbizopoulos/comprehensive-comparison-of-deep-lear"
        "ning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/"
        "default/tmp"
    )
    if output != subject.RUNTIME_DIR:
        message = "Runtime output must use the canonical package directory"
        raise AssertionError(message)
    subject.main([])
    with (output / "keys-values.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not row.get("key") or not row.get("value") for row in rows):
        msg = "the results must contain named, populated measurements"
        raise AssertionError(msg)
    if len({row["key"] for row in rows}) != len(rows):
        msg = "result names must be unique"
        raise AssertionError(msg)
    manuscript = (output / "ms.pdf").read_bytes()
    if not manuscript.startswith(b"%PDF-") or not manuscript.rstrip().endswith(
        b"%%EOF",
    ):
        msg = "the manuscript must be a complete PDF"
        raise AssertionError(msg)
