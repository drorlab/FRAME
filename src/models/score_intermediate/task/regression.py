import collections as col
import pandas as pd
import wandb

import torch
import torch.nn as nn
import torchmetrics

from src.models.score_intermediate.task.base import Task


class RegressionTask(Task):
    def __init__(self):
        super().__init__()

        # Loss function
        self.loss_fn = nn.functional.smooth_l1_loss
        # Metrics
        self.pl_metrics = {
            'train': self._create_pl_metrics(),
            'val': self._create_pl_metrics(),
            'test': self._create_pl_metrics(),
        }

    def _create_pl_metrics(self):
        return nn.ModuleDict({
            'mse': torchmetrics.MeanSquaredError(),
            })

    def compute_pl_metrics(self, mode, prefix, suffix,
                           compute_only=True, y_hat=None, target=None):
        return self.compute_pl_metrics(prefix, suffix, self.pl_metrics[mode],
                                       compute_only, y_hat, target)

    @staticmethod
    def compute_epoch_metrics(df, current_epoch, prefix, suffix, scalar, plot):
        if len(df) < 1:
            return {}
        y_true = df['target'].astype(float)
        y_hat = df['y_hat'].astype(float)
        mse = ((y_true - y_hat)**2).mean()
        metrics[f'{prefix}mse{suffix}'] = mse
        metrics[f'{prefix}rmse{suffix}'] = mse**0.5
        # Skip if only 2 data points or less since the correlations are
        # not really meaningful.
        if len(y_true) > 2:
            metrics[f'{prefix}pearson{suffix}'] = y_true.corr(y_hat, method='pearson')
            metrics[f'{prefix}spearman{suffix}'] = y_true.corr(y_hat, method='spearman')
            metrics[f'{prefix}kendall{suffix}'] = y_true.corr(y_hat, method='kendall')
        return metrics

    def loss(self, y_hat, labels):
        return self.loss_fn(input=y_hat, target=labels)

    def predict(self, y_hat):
        return y_hat

    def collate_result(self, current_epoch, batch_idx, ids, labels, y_hat):
        data = col.defaultdict(list)
        data['id'].extend(ids)
        data['target'].extend(labels.cpu().numpy())
        data['y_hat'].extend(y_hat.cpu().numpy())

        df = pd.DataFrame(data)
        df.insert(0, 'batch_idx', batch_idx)
        df.insert(0, 'epoch', current_epoch)
        return df
