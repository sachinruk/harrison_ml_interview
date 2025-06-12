import torch
import torchmetrics


class AveragePrecision(torchmetrics.Metric):
    """Computes Precision per Image then averages over all images."""

    def __init__(self, trimap_threshold: float = 0.9, threshold: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.trimap_threshold = trimap_threshold
        self.threshold = threshold
        self.add_state("precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, trimap: torch.Tensor):
        preds = preds > self.threshold
        target = trimap > self.trimap_threshold
        tp = (preds * target).sum(dim=(1, 2, 3))  # True Positives
        predicted_positives = preds.sum(dim=(1, 2, 3))
        precision = tp / (predicted_positives + 1e-8)
        self.precision_sum += precision.sum()
        self.n += len(precision)

    def compute(self):
        return self.precision_sum / (self.n + 1e-8)  # Avoid division by zero


class AverageRecall(torchmetrics.Metric):
    """Computes Recall per Image then averages over all images."""

    def __init__(self, trimap_threshold: float = 0.9, threshold: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.trimap_threshold = trimap_threshold
        self.threshold = threshold
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, trimap: torch.Tensor):
        preds = preds > self.threshold
        target = trimap > self.trimap_threshold
        tp = (preds * target).sum(dim=(1, 2, 3))  # True Positives
        actual_positives = target.sum(dim=(1, 2, 3))
        recall = tp / (actual_positives + 1e-8)
        self.recall_sum += recall.sum()
        self.n += len(recall)

    def compute(self):
        return self.recall_sum / (self.n + 1e-8)  # Avoid division by zero
