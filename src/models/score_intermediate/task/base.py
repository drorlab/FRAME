import pandas as pd
from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self):
        # For saving validation and prediction results
        self.train_results = pd.DataFrame()
        self.val_results = pd.DataFrame()
        self.test_results = pd.DataFrame()

    @staticmethod
    def compute_pl_metrics(prefix, suffix, metrics_dict, compute_only=True,
                           preds=None, target=None):
        metrics = {}
        for name, metric_fn in metrics_dict.items():
            if compute_only:
                assert (not preds) and (not target)
                score = metric_fn.compute()
            else:
                score = metric_fn(preds, target)
            metrics[f'{prefix}{name}{suffix}'] = score
        return metrics

    @abstractmethod
    def compute_epoch_metrics(df, current_epoch, prefix, suffix, scalar, plot):
        raise NotImplementedError

    @abstractmethod
    def loss(self, y_hat, labels):
        raise NotImplementedError

    @abstractmethod
    def predict(self, y_hat):
        raise NotImplementedError

    @abstractmethod
    def collate_result(self, current_epoch, batch_idx, ids, labels, y_hat):
        raise NotImplementedError
