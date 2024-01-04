import argparse as ap
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sklearn.metrics as sm

from atom3d.datasets import LMDBDataset
import dotenv as de
import wandb

import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as log
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import src.models.score_intermediate.model as m
import src.models.score_intermediate.metrics as met
from src.models.score_intermediate.data import ScoreFragmentModelDataModule
from src.models.score_intermediate.util.config import defaults
from src.models.score_intermediate.util.data_transform import calculate_pos_weight

de.load_dotenv(de.find_dotenv(usecwd=True))
logger = logging.getLogger("lightning")


def setup_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('--group_name', type=str, default=defaults['group_name'])
    parser.add_argument('--name', type=str)
    parser.add_argument('--project_name', type=str, default=defaults['project_name'])
    parser.add_argument('--run_id', type=str, default=defaults['run_id'])
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # Whether it is a binary classification or regression task
    parser.add_argument('--task', type=str, default=defaults['task'],
                        choices=['binary', 'regression'])
    # Whether is using pdbbind dataset or fragment_stitching
    parser.add_argument('--dataset', type=str, default=defaults['dataset'],
                        choices=['fragment', 'fragment_avg_ligand', 'fragment_avg_fragment',
                                 'pdbbind', 'rmsd', 'norm_rmsd', 'per_atom', 'frag_type'])
    # Add PROGRAM level args
    parser.add_argument('-train', '--train_dataset', type=str, nargs='*',
                        default=[defaults['train_dataset']])
    parser.add_argument('-val', '--val_dataset', type=str, nargs='*',
                        default=[defaults['val_dataset']])
    parser.add_argument('-test', '--test_dataset', type=str, nargs='*',
                        default=[defaults['test_dataset']])
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'])
    parser.add_argument('--num_workers', type=int, default=defaults['num_workers'])
    parser.add_argument('--early_stopping', default=defaults['early_stopping'],
                        action='store_true')
    # Add data transform related flags
    parser.add_argument('--random_seed', '-seed', type=int, default=defaults['random_seed'])
    parser.add_argument('--element_set', '-el', type=str, default=defaults['element_set'],
                        choices=['group', 'HCONF'])
    parser.add_argument('--use_dummy', default=defaults['use_dummy'],
                        action='store_true',
                        help="Map unrecognized element to dummy variable")
    parser.add_argument('--no_ligand_flag', action='store_true',
                        default=defaults['no_ligand_flag'])
    parser.add_argument('--flag_fragment', action='store_true',
                        default=defaults['flag_fragment'])
    parser.add_argument('--sample_pocket_atoms_frac', '-pfrac', type=float,
                        default=defaults['sample_pocket_atoms_frac'],
                        help="If less than 1.0, randomly sample atom pockets to use as model input")
    # Binary task related flags
    parser.add_argument('--balance_dataset', '-balance',
                        default=defaults['balance_dataset'], action='store_true')
    parser.add_argument('--upweight_pos', '-up', default=defaults['upweight_pos'],
                        action='store_true')
    parser.add_argument('--label_noise_sigma', type=float,
                        default=defaults['label_noise_sigma'],
                        help="Add gaussian random noise between [-label_noise_sigma, label_noise_sigma] to label. "
                             "Only works for regression type learning")
    parser.add_argument('--agg_focus', action='store_true', default=defaults['agg_focus'])
    parser.add_argument('--weighted_loss', action='store_true', default=defaults['weighted_loss'])
    parser.add_argument('--pretrained_ckpt_path', type=str, default=defaults['pretrained_ckpt_path'])
    # Add model specific args
    parser = m.ScoreFragmentModel.add_model_specific_args(parser)
    # Add trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    ## Note: to resume training from specific checkpoint, pass in the
    # --resume_from_checkpoint flag.
    #
    ## Note: for large datasets, we might want to check validation multiple times
    # within a training loop by passing in the --val_check_interval flag.
    # Pass in a float to check that often within 1 training epoch.
    # Pass in an int k to check every k training batches.
    return parser


def check_flags(hparams):
    #TODO(psuriana): Implement this
    pass


def setup_run(hparams):
    run = wandb.init(
        group=hparams.group_name,
        project=hparams.project_name,
        id=hparams.run_id,
        dir=os.environ['MODEL_DIR'],
        name=hparams.run_id,
        config=hparams,
        settings=wandb.Settings(start_method="fork"),
        )
    return run


def main():
    parser = setup_parser()
    hparams = parser.parse_args()
    hparams.cuda = not hparams.no_cuda and torch.cuda.is_available()
    check_flags(hparams)
    # Set logging format
    logging.basicConfig(stream=sys.stdout,
                        format=f'%(asctime)s %(levelname)s %(process)d: %(message)s',
                        level=logging.INFO)
    # Initialize wandb
    run = setup_run(hparams)
    # Add wandb output dir
    hparams.output_dir = run.dir
    # Run train/val/test
    train(hparams)


def train(hparams):
    dict_args = vars(hparams)

    # Need to set the random seed to ensure each copy of the model on the GPUs
    # starts from the same state if we are running distributed training
    if hparams.random_seed == None:
        hparams.random_seed = int(np.random.randint(1, 10e6))

    logger.info(f"Set random seed to {hparams.random_seed:}...")
    pl.seed_everything(hparams.random_seed, workers=True)

    # Setup data
    logger.info(f"Setup dataloaders...")
    data_module = ScoreFragmentModelDataModule(use_labels=True, **dict_args)
    # Update hparams with input data relevant dims
    dict_args.update(data_module.transform.get_dims())

    # Initialize logger
    logger.info(f"Logging wandb to project {hparams.project_name:}...")
    wandb_logger = log.WandbLogger(save_dir=hparams.output_dir,
                                   log_model=True)

    # Initialize model

    #upweight positive examples if desired
    if hparams.upweight_pos:
        hparams.pos_weight = calculate_pos_weight(LMDBDataset(hparams.train_dataset))
        logger.info(f"Positive weight of training dataset: {hparams.pos_weight:}")
    logger.info("Initializing model...")

    if hparams.pretrained_ckpt_path == '':
        model = m.ScoreFragmentModel(**dict_args)
    else:
        print("LOADING MODEL from ", hparams.pretrained_ckpt_path)
        model = m.ScoreFragmentModel.load_from_checkpoint(hparams.pretrained_ckpt_path)
    if hparams.weighted_loss:
        model.update_param("weighted_loss", True)
        model.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Log model topology
    print(model.net)
    wandb_logger.watch(model.net, log='all')

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(hparams.output_dir, 'checkpoints'),
        filename='FragS-{epoch:03d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        #save_weights_only=True,
        )
    callbacks = [checkpoint_callback]
    if hparams.lr_schedule == 'cosine':
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    if hparams.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=False,
            mode='min',
            )
        callbacks.append(early_stop_callback)

    logger.info(f"Saving output to {hparams.output_dir:}...")
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        accelerator='gpu',
        devices=(1 if hparams.cuda else 0),
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        #deterministic=True,
        )

    # TRAINING
    logger.info(f"Running training on {hparams.train_dataset:} with val {hparams.val_dataset:}...")
    out = trainer.fit(model, data_module)

    # SAVE RESULTS
    # Training results
    train_filename = os.path.join(hparams.output_dir, 'train_results.csv')
    logger.info(f"Save training results to {train_filename:}...")
    model.train_results.to_csv(train_filename, index=False, float_format='%.7f')
    # Validation results
    val_filename = os.path.join(hparams.output_dir, 'val_results.csv')
    logger.info(f"Save validation results to {val_filename:}...")
    model.val_results.to_csv(val_filename, index=False, float_format='%.7f')

    val_df = model.val_results
    if len(val_df) > 0:
        #val_df = model.val_results[model.val_results['epoch'] == model.current_epoch]
        targets, preds = val_df['target'], val_df['pred']
        # Log to wandb
        metrics = met.compute_metrics(
            hparams.task, targets, preds, model.current_epoch, 'val/', '')
        metrics['preds/val'] = wandb.Table(dataframe=val_df)
        wandb_logger.log_metrics(metrics)

    # TESTING
    if hparams.test_dataset:
        logger.info(f"Running test on {hparams.test_dataset:}...")
        out = trainer.test(model, datamodule=data_module)
        # Save training results
        test_filename = os.path.join(hparams.output_dir, 'test_results.csv')
        test_df = model.test_results
        if len(test_df) > 0:
            logger.info(f"Save test results to {test_filename:}...")
            test_df.to_csv(test_filename, index=False, float_format='%.7f')

            targets, preds = test_df['target'], test_df['pred']
            # Log to wandb
            metrics = met.compute_metrics(
                hparams.task, targets, preds, model.current_epoch, 'test/', '')
            metrics['preds/test'] = wandb.Table(dataframe=test_df)
            wandb_logger.log_metrics(metrics)

            print("--------------------------------------------------------------------------------")
            print("Test results:")
            if hparams.task == 'binary':
                binary_preds = (preds > hparams.pos_threshold).astype(int)
                print(sm.classification_report(targets, binary_preds, target_names=['neg', 'pos']))
                kappa = sm.cohen_kappa_score(targets, binary_preds, weights='quadratic')
                print(f"kappa: {kappa:.4f}")
            elif hparams.task == 'regression':
                metrics = met.regression_metrics(targets, preds)
                print(f"RMSE: {metrics['rmse']:.4f}, Pearson: {metrics['pearson']:.4f}, "
                      f"Spearman: {metrics['spearman']:.4f}")
            print("--------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
