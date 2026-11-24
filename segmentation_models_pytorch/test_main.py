# Copyright (c) 2026- Paschalis Bizopoulos
"""Check the segmentation loss used by the paper."""

import torch
from segmentation_models_pytorch.utils.losses import DiceLoss


def test_dice_loss_is_zero_for_matching_masks_and_positive_for_disjoint_masks() -> None:
    """Exercise the same Dice loss API imported by the training code."""
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = DiceLoss()
    matched = loss(target, target).item()
    disjoint = loss(1 - target, target).item()
    if matched != 0 or disjoint <= 0:
        raise AssertionError((matched, disjoint))
