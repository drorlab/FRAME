import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

import argparse as ap
import collections as col
import pandas as pd
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from src.models.score_intermediate.enn.model import E3NN_Model, E3NN_Model_Per_Atom, E3NN_Model_Frag_Type
from src.models.score_intermediate.cnn3d.model import CNN3D_Model
from src.models.score_intermediate.util.config import defaults

from abc import ABC, abstractmethod

class BaseModel(pl.LightningModule):

    METRICS_SUFFIX = "_metrics"
    TRAINING = "train"
    TESTING = "test"
    VALIDATION = "val"

    def __init__(self, learning_rate=1e-4, pos_threshold=0.5,
                 pos_weight=None, **kwargs):
        super().__init__()

        # Save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        # To store results
        self.train_results = pd.DataFrame()
        self.val_results = pd.DataFrame()
        self.test_results = pd.DataFrame()

        # Network
        self.net = None

        # Metrics
        self.metrics = None

        # Method to compute predictions and loss
        self.get_preds_loss = None

    def get_progress_bar_dict(self):
        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.lr_schedule == 'constant':
            print("Use constant LR schedule")
            return optimizer
        elif self.hparams.lr_schedule == 'cosine':
            print("Use CosineAnnealing LR schedule")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Invalid LR schedule {self.hparams.lr_schedule:}")

    def forward(self, d):
        return self.net(d)

    def training_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.get_loss(batch, batch_idx)

    def get_loss(self, batch, batch_idx):
         preds, loss = self.get_preds_loss(batch, batch_idx)
         targets = batch['label'].int()
         ids = np.array(batch['id'])
         #TODO update for frag_type
         if self.hparams.dataset == 'per_atom':
             sel = torch.squeeze(batch.batch[batch.select_atoms_index]).cpu().numpy()
             preds = torch.squeeze(preds)
             targets = torch.squeeze(targets)
             ids = ids[sel]
         res = {
             'preds': preds.detach(),
             'loss': loss,
             'targets': targets,
             'ids': ids,
             'batch_idx': batch_idx,
             }
         return res

    def training_step_end(self, outputs):
        # Update and log metrics
        self.update_metrics(self.TRAINING, outputs["preds"], outputs["targets"])

    def validation_step_end(self, outputs):
        # Update and log metrics
        self.update_metrics(self.VALIDATION, outputs["preds"], outputs["targets"])

    def test_step_end(self, outputs):
        # Update and log metrics
        self.update_metrics(self.TESTING, outputs["preds"], outputs["targets"])

    def training_epoch_end(self, outputs):
        df = self.collate_result(outputs)
        # Keep only the last training epoch results
        #self.train_results = df
        self.train_results = pd.concat([self.train_results, df], ignore_index=True)

    def validation_epoch_end(self, outputs):
        df = self.collate_result(outputs)
        self.val_results = pd.concat([self.val_results, df], ignore_index=True)

    def test_epoch_end(self, outputs):
        df = self.collate_result(outputs)
        self.test_results = pd.concat([self.test_results, df], ignore_index=True)

    def collate_result(self, outputs):
        losses = []
        data = col.defaultdict(list)
        for i, x in enumerate(outputs):
            losses.append(x['loss'])
            data['rank'].extend([i]*len(x['ids']))
            data['batch_idx'].extend([x['batch_idx']]*len(x['ids']))
            data['id'].extend(x['ids'])
            data['target'].extend(x['targets'].cpu().numpy())
            data['pred'].extend(x['preds'].cpu().numpy())
        avg_loss = torch.stack(losses).mean()
        self.log('val_loss', avg_loss, prog_bar=True)

        df = pd.DataFrame(data)
        df.insert(0, 'epoch', self.current_epoch)
        return df

    @staticmethod
    def _get_metric_name(m: torchmetrics.Metric):
        return m.__class__.__name__.lower()

    def update_metrics(self,
                       stage,
                       preds: torch.Tensor,
                       targets: torch.Tensor,
                       log_metrics=True):
        # Update and log metrics
        for metric in self.metrics[stage + self.METRICS_SUFFIX]:
            m = metric(preds, targets)
            # TODO(psuriana): figure out how to log tuple
            if log_metrics and (type(m) is not tuple):
                self.log(
                    f"{stage}/{self._get_metric_name(metric)}",
                    m,
                    on_step=False,
                    on_epoch=True,
                )


class ScoreFragmentModel(BaseModel):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=defaults['learning_rate'])
        parser.add_argument('--lr_schedule', choices=['constant', 'cosine'],
                            type=str.lower, default=defaults['lr_schedule'])
        parser.add_argument('--pos_threshold', type=float, default=defaults['pos_threshold'])
        # Which model to use
        parser.add_argument('--model', type=str, default=defaults['model'],
                            choices=['e3nn', 'cnn3d'])
        # E3NN model params
        parser.add_argument('--rbf_high', type=float, default=defaults['rbf_high'])
        parser.add_argument('--rbf_count', type=int, default=defaults['rbf_count'])
        parser.add_argument('--num_nearest_neighbors', type=int, default=defaults['num_nearest_neighbors'])
        parser.add_argument('--e3nn_filters', nargs='*', type=int, default=defaults['e3nn_filters'])
        parser.add_argument('--fc_filters', nargs='*', type=int, default=defaults['fc_filters'])
        # 3DCNN model params
        parser.add_argument('--conv_drop_rate', type=float, default=defaults['conv_drop_rate'])
        parser.add_argument('--fc_drop_rate', type=float, default=defaults['fc_drop_rate'])
        parser.add_argument('--batch_norm', action='store_true', default=defaults['batch_norm'])
        parser.add_argument('--no_dropout', action='store_true', default=defaults['no_dropout'])
        parser.add_argument('--num_conv', type=int, default=defaults['num_conv'])
        parser.add_argument('--conv_kernel_size', type=int, default=defaults['conv_kernel_size'])
        parser.add_argument('--fc_units', nargs='*', type=int, default=defaults['fc_units'])
        return parser

    def __init__(self, learning_rate=1e-4, pos_threshold=0.5,
                 pos_weight=None, **kwargs):
        super().__init__(learning_rate, pos_threshold, pos_weight, **kwargs)

        self.save_hyperparameters()

        # Define network
        if self.hparams.model == 'e3nn':
            if self.hparams.dataset == 'per_atom':
                self.net = E3NN_Model_Per_Atom(**self.hparams)
            elif self.hparams.dataset == 'frag_type':
                self.net = E3NN_Model_Frag_Type(**self.hparams)
            else:
                self.net = E3NN_Model(**self.hparams)
        elif self.hparams.model == 'cnn3d':
            self.net = CNN3D_Model(dropout=not self.hparams.no_dropout,
                                   **self.hparams)
        else:
            raise ValueError(f"Invalid model option {self.hparams.model:}")

        # Set loss function and metrics based on the task type
        if self.hparams.task == 'regression':
            # Initialize loss function
            self.loss_fn = nn.functional.smooth_l1_loss
            self.get_preds_loss = self._regression_preds_loss
            # Metrics
            _metrics_factory = lambda: nn.ModuleList([
                torchmetrics.MeanSquaredError(dist_sync_on_step=True),
                #torchmetrics.PearsonCorrcoef(dist_sync_on_step=True),
                #torchmetrics.SpearmanCorrcoef(dist_sync_on_step=True),
                ])
        elif self.hparams.task == 'binary':
            self.weighted_loss = False
            # Initialize loss function
            if self.hparams.upweight_pos:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_weight,
                                                    reduction='mean')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
            self.get_preds_loss = self._binary_preds_loss
            # Metrics
            _metrics_factory = lambda: nn.ModuleList([
                #torchmetrics.Accuracy(task='binary', threshold=self.hparams.pos_threshold, dist_sync_on_step=True),
                #torchmetrics.Precision(task='binary', threshold=self.hparams.pos_threshold, dist_sync_on_step=True),
                #torchmetrics.Recall(task='binary', threshold=self.hparams.pos_threshold, dist_sync_on_step=True),
                torchmetrics.Accuracy(threshold=self.hparams.pos_threshold, dist_sync_on_step=True),
                torchmetrics.Precision(threshold=self.hparams.pos_threshold, dist_sync_on_step=True),
                torchmetrics.Recall(threshold=self.hparams.pos_threshold, dist_sync_on_step=True),
                ])
        else:
            raise ValueError(f"Invalid task {self.hparams.task:}")

        # Initialize metrics
        self.metrics = nn.ModuleDict({
            self.TRAINING + self.METRICS_SUFFIX: _metrics_factory(),
            self.VALIDATION + self.METRICS_SUFFIX: _metrics_factory(),
            self.TESTING + self.METRICS_SUFFIX: _metrics_factory(),
            })

    def update_param(self, name, value):
        setattr(self, name, value)

    ############################### Binary task related ###############################
    def _binary_preds_loss(self, batch, batch_idx):
        logits = self(batch)
        if self.weighted_loss:
            loss = torch.mean(self.loss_fn(input=logits, target=batch['label'])*batch['weight'])
        else:
            loss = self.loss_fn(input=logits, target=batch['label'])
        preds = torch.sigmoid(logits)
        return preds, loss

    ############################### Regression task related ###############################
    def _regression_preds_loss(self, batch, batch_idx):
        targets = batch['label'].float()
        preds = self(batch)
        loss = self.loss_fn(input=preds, target=targets)
        return preds, loss
