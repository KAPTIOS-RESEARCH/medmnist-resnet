import torch
from src.utils.config import instanciate_module
from src.utils.device import get_available_device

class Evaluator:
    def __init__(self, metrics: dict):

        self.device = get_available_device()
        self.metrics = {}

        for metric in metrics:
            metric_instance = instanciate_module(
                metric['module_name'], metric['class_name'], metric['parameters'])
            self.metrics[metric['class_name']] = metric_instance.to(self.device)

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        if not self.metrics:
            return {}

        preprocess_map = {
            'Accuracy': lambda preds, targets:
                (preds.squeeze() if preds.shape[1] ==
                 1 else preds.argmax(dim=1), targets),
            'AUROC': lambda preds, targets:
                (preds.squeeze() if preds.shape[1] ==
                 1 else preds.softmax(dim=1), targets),
        }

        results = {}
        for name, metric in self.metrics.items():
            if name in preprocess_map:
                processed_preds, processed_targets = preprocess_map[name](
                    preds, targets)
                results[name] = metric(processed_preds, processed_targets)
            else:
                results[name] = metric(preds, targets)
        return results
