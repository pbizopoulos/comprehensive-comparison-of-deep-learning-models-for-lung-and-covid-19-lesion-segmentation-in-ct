# Copyright (c) 2026- Paschalis Bizopoulos
"""Check the segmentation loss and the generated manuscript."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import generate_paper

if TYPE_CHECKING:
    import pytest


_EXPECTED_MS_PDF_SHA256 = (
    "2731eaa036389a3ae737db1f4728822b64a33df14bcad271655c675f55d6e7a0"
)


def test_generated_manuscript_pdf_matches_the_expected_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate the manuscript and compare its bytes with the pinned baseline."""
    output = tmp_path / generate_paper.RUNTIME_DIR.relative_to(Path.home())
    monkeypatch.setattr(generate_paper, "RUNTIME_DIR", output)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    generate_paper.main(smoke=True)
    repository = (
        "comprehensive-comparison-of-deep-learning-models-for-"
        "lung-and-covid-19-lesion-segmentation-in-ct"
    )
    pdf = tmp_path / "github.com/pbizopoulos" / repository / "generate_paper/tmp/ms.pdf"
    actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if actual != _EXPECTED_MS_PDF_SHA256:
        message = f"ms.pdf SHA-256: expected {_EXPECTED_MS_PDF_SHA256}, got {actual}"
        raise AssertionError(message)


def test_soft_dice_pools_an_empty_and_a_nonempty_mask_in_one_batch() -> None:
    """Keep the smoothing and the gradient for predictions on background pixels."""
    prediction = torch.tensor([[0.75, 0.25], [0.5, 0.5]], requires_grad=True)
    target = torch.tensor([[True, False], [False, False]])
    loss = generate_paper.soft_dice_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(0.375))
    (gradient,) = torch.autograd.grad(loss, prediction)
    torch.testing.assert_close(
        gradient, torch.tensor([[-0.34375, 0.15625], [0.15625, 0.15625]])
    )
