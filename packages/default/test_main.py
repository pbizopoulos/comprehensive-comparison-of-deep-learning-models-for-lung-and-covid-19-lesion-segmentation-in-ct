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
    monkeypatch.chdir(tmp_path)
    from packages.default import main as subject  # noqa: PLC0415

    subject.main()
    with (tmp_path / "tmp/keys-values.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not row.get("key") or not row.get("value") for row in rows):
        msg = "the results must contain named, populated measurements"
        raise AssertionError(msg)
    if len({row["key"] for row in rows}) != len(rows):
        msg = "result names must be unique"
        raise AssertionError(msg)
    manuscript = (tmp_path / "tmp/ms.pdf").read_bytes()
    if not manuscript.startswith(b"%PDF-") or not manuscript.rstrip().endswith(
        b"%%EOF",
    ):
        msg = "the manuscript must be a complete PDF"
        raise AssertionError(msg)
