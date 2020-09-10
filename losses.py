eps = 1e-6

def dice_loss(prediction, mask):
    tp = (mask * prediction).sum([1, 2, 3])
    fp = (mask * (1 - prediction)).sum([1, 2, 3])
    fn = (~mask * prediction).sum([1, 2, 3])
    is_mask_and_prediction_empty = (tp + fp) == 0
    dice_loss_ = 1 - 2*tp / (2*tp + fp + fn + eps)
    dice_loss_ *= ~is_mask_and_prediction_empty
    return dice_loss_.mean()

def metrics(prediction, mask):
    prediction = prediction > 0.5
    tp = (mask * prediction).sum().float()
    tn = (~mask * ~prediction).sum().float()
    fp = (mask * ~prediction).sum().float()
    fn = (~mask * prediction).sum().float()
    is_mask_and_prediction_empty = (tp + fp) == 0
    if is_mask_and_prediction_empty:
        specificity = 1
        sensitivity = 1
        dice = 1
    else:
        specificity = (tn / (tn + fp + eps)).item()
        sensitivity = (tp / (tp + fn + eps)).item()
        dice = (2*tp / (2*tp + fp + fn + eps)).item()
    return sensitivity, specificity, dice
