import torch
from torchmetrics.functional import (
    normalized_root_mean_squared_error,
    pearson_corrcoef,
)


def compute_metrics(preds, target):
    nrmse = normalized_root_mean_squared_error(preds, target, normalization="range")
    r2 = pearson_corrcoef(preds, target) ** 2
    final_score = 0.4 * (1 - torch.clamp(nrmse, max=1.0)) + 0.6 * r2
    return {
        "score": final_score,
        "nrmse": nrmse,
        "r2": r2,
    }
