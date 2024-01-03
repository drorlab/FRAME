import argparse as ap
import logging
import os
import pathlib
import sys

import dotenv as de
import pandas as pd
import torch_geometric
import wandb

import pytorch_lightning as pl
import pytorch_lightning.loggers as log

from src.models.score_intermediate.data import ScoreFragmentModelDataModule
import src.models.score_intermediate.model as m
from src.models.score_intermediate.util.config import defaults

de.load_dotenv(de.find_dotenv(usecwd=True))
logger = logging.getLogger("lightning")


def main():
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--project_name', type=str, default='fragment_stitching')
    parser.add_argument('--run_id', type=str, default='train')
    parser.add_argument('-test', '--test_dataset', type=str,
                        default=defaults['test_dataset'])
    # add PROGRAM level args
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--nolabels', dest='use_labels', action='store_false')
    # Add data transform related flags
    parser.add_argument('--random_seed', '-seed', type=int, default=None)

    # Add model specific args
    parser = m.ScoreFragmentModel.add_model_specific_args(parser)

    # Add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    # Load model from checkpoint
    logger.info("Loading model weights...")
    model = m.ScoreFragmentModel.load_from_checkpoint(hparams.checkpoint_path)
    # Update hparams with model, task, and data transform related flags
    # the model was trained with
    for key in ['model', 'task', 'dataset',
                'element_set', 'use_dummy', 'no_ligand_flag', 'flag_fragment']:
        dict_args[key] = model._hparams[key] if key in model._hparams else defaults[key]

    # Initialize wandb
    logger.info(f"Logging wandb to project {hparams.project_name:}...")
    wandb.init(project=hparams.project_name, dir=os.environ['MODEL_DIR'],
               name=hparams.run_id, config=hparams)

    # Setup data
    logger.info(f"Setup dataloaders...")
    data_module = ScoreFragmentModelDataModule(**dict_args)
    # Update hparams with input data relevant dims
    dict_args.update(data_module.transform.get_dims())

    # Initialize logger
    logger.info(f"Logging wandb to project {hparams.project_name:}...")
    wandb_logger = log.WandbLogger(save_dir=os.environ['MODEL_DIR'],
                                   log_model=True)

    # Initialize trainer
    logger.info(f"Saving wandb run to {wandb.run.dir:}...")
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        logger=wandb_logger,
        log_every_n_steps=1)

    # Prediction
    logger.info(f"Running prediction for dataset {hparams.dataset:} with {hparams.model:} model...")
    out = trainer.test(model, datamodule=data_module, verbose=False)

    # Save output
    test_filename = os.path.join(wandb.run.dir, 'test_results.csv')
    logger.info(f"Save test results to {test_filename:}...")
    test_df = model.task.test_results
    test_df.to_csv(test_filename, index=False, float_format='%.7f')
    # Hack to log plot-related test results (NOTE: for some reasons, if this is done
    # on test_epoch_end, it will throw "cannot pickle '_thread.lock' object"
    # during deepcopy of the trainer.logger_connector.callback_metrics)
    metrics = model.task.compute_epoch_metrics(
        test_df, model.current_epoch, 'test/', '_sk_epoch', scalar=False, plot=True)
    metrics['preds/test'] = wandb.Table(dataframe=test_df)
    wandb_logger.log_metrics(metrics)

    print("--------------------------------------------------------------------------------")
    print("Test results:")
    if hparams.task == 'binary':
        print(sm.classification_report(test_df['target'], test_df['pred'], target_names=['neg', 'pos']))
        kappa = sm.cohen_kappa_score(test_df['target'], test_df['pred'], weights='quadratic')
        print(f"kappa: {kappa:.4f}")
    elif hparams.task == 'regression':
        metrics = model.task.compute_epoch_metrics(test_df, '', '', '', scalar=False, plot=True)
        print(f"RMSE: {metrics['rmse']:.4f}, Pearson: {metrics['pearson']:.4f}, "
              f"Spearman: {metrics['spearman']:.4f}")
    print("--------------------------------------------------------------------------------")

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()
