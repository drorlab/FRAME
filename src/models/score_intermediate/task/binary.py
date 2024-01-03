import collections as col
import wandb

import numpy as np
import pandas as pd
import sklearn.metrics as sm

import torch
import torch.nn as nn
import torchmetrics

from src.models.score_intermediate.task.base import Task


class BinaryTask(Task):
    def __init__(self, pos_threshold=0.5, upweight_pos=False, pos_weight=None):
        super().__init__()

        self.pos_threshold = pos_threshold
        # Loss function
        if upweight_pos:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        # Metrics
        self.pl_metrics = {
            'train': self._create_pl_metrics('train'),
            'val': self._create_pl_metrics('val'),
            'test': self._create_pl_metrics('test'),
        }

    def _create_pl_metrics(self, mode):
        return nn.ModuleDict({
            # Pytorch lightning default is macro average
            ### Global metrics
            'acc': torchmetrics.Accuracy(threshold=self.pos_threshold),
            ### Metrics for positive class
            'prec': torchmetrics.Precision(threshold=self.pos_threshold),
            #'recall': torchmetrics.Recall(threshold=self.pos_threshold),
            #'f1': torchmetrics.F1(threshold=self.pos_threshold),
            })

    def compute_pl_metrics(self, mode, prefix, suffix,
                           compute_only=True, y_hat=None, target=None):
        preds =  y_hat if compute_only else self.predict(y_hat)[0]
        target = target if compute_only else target.int()
        return Task.compute_pl_metrics(prefix, suffix, self.pl_metrics[mode],
                                       compute_only, preds, target)

    @staticmethod
    def compute_epoch_metrics(df, current_epoch, prefix, suffix, scalar, plot):
        if len(df) < 1:
            return {}
        metrics = {}
        y_true = df['target']
        y_prob = df['y_prob']
        ### Scalar metrics: AUROC etc2
        if scalar:
            metrics[f'{prefix}auroc{suffix}'] = sm.roc_auc_score(y_true, y_prob)
            metrics[f'{prefix}auprc{suffix}'] = sm.average_precision_score(y_true, y_prob)
            #metrics[f'{prefix}acc{suffix}'] = sm.accuracy_score(y_true, y_prob.round())
        ### Plot metrics
        if plot:
            # Confusion matrix
            metrics[f'{prefix}conf_mat{suffix}'] = wandb.plot.confusion_matrix(
                y_true=df['target'].values, preds=df['y_pred'].values, class_names=['neg', 'pos'],
                title=f'{prefix} epoch {current_epoch}')
            # Precision-recall and ROC curves
            y_prob = np.vstack([1-df['y_prob'], df['y_prob']]).T # [neg_prob, pos_prob]
            metrics[f'{prefix}pr_curve{suffix}'] = wandb.plot.pr_curve(
                y_true, y_prob, labels=['neg', 'pos'],
                title=f'{prefix} epoch {current_epoch}: Precision v. Recall')
            metrics[f'{prefix}roc{suffix}'] = wandb.plot.roc_curve(
                y_true, y_prob, labels=['neg', 'pos'],
                title=f'{prefix} epoch {current_epoch}: ROC')
        return metrics

    def loss(self, y_hat, labels):
        return self.loss_fn(input=y_hat, target=labels)

    def predict(self, y_hat):
        y_probs = torch.sigmoid(y_hat)
        # Predict positive (1) if prob > threshold
        y_preds = (y_probs > self.pos_threshold).int()
        return y_probs, y_preds

    def collate_result(self, current_epoch, batch_idx, ids, labels, y_hat):
        data = col.defaultdict(list)
        data['id'].extend(ids)
        data['target'].extend(labels.cpu().numpy())
        data['y_hat'].extend(y_hat.cpu().numpy())

        # Convert to probabilities and binary class predictions (0/1)
        y_probs, y_preds = self.predict(y_hat)
        data['y_prob'].extend(y_probs.cpu().numpy())
        data['y_pred'].extend(y_preds.cpu().numpy())

        df = pd.DataFrame(data)
        df.insert(0, 'batch_idx', batch_idx)
        df.insert(0, 'epoch', current_epoch)
        return df
