# Copyright (c) 2024 Panagiotis Bizopoulos
"""Tests for default."""

from __future__ import annotations

from packages.default.main import (
    _OUT_PATH,
    main,
)


def test_main() -> None:
    """Generate the test artifacts and compile the manuscript."""
    main()
    assert (_OUT_PATH / "keys-values.csv").is_file()  # noqa: S101
    assert (_OUT_PATH / "ms.pdf").is_file()  # noqa: S101
