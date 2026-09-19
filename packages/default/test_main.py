# Copyright (c) 2026- Paschalis Bizopoulos
"""Specify the installed research pipeline's offline example workflow."""

from __future__ import annotations

import csv
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_smoke_run_generates_readable_results_and_a_complete_manuscript(
    tmp_path: Path,
) -> None:
    """The installed command produces CSV results and a compiled PDF offline."""
    result = subprocess.run(  # noqa: S603
        [os.environ["PACKAGE_E2E_EXECUTABLE"], "--smoke"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
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
