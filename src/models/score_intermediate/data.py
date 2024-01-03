import os
import os.path as osp
from operator import itemgetter
import logging
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.utils.data as td

import torch_geometric
import torch_geometric.data as tgd

import pytorch_lightning as pl

from torch.utils.data import DistributedSampler
from torch.utils.data.dataset import ConcatDataset

from atom3d.datasets import LMDBDataset

from src.models.score_intermediate.cnn3d.data import CNN3D_Transform
from src.models.score_intermediate.enn.data import E3NN_Transform
from src.models.score_intermediate.util.data_transform import *

logger = logging.getLogger("lightning")


class DatasetFromSampler(td.Dataset):
    """
    From:
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/dataset/torch.py

    Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    From:
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py#L683

    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True, seed=None):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
        )
        self.sampler = sampler

    def __iter__(self):
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class FragmentDataset(tgd.Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.lmdb_dataset = LMDBDataset(root)
        super(FragmentDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> str:
        return 'data.mdb'

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self))]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root)

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', repr(self.pre_transform))
        #return osp.join('tmp', 'processed', repr(self.pre_transform))

    def process(self):
        """Processes raw data and saves it into the processed_dir."""
        # Read data from lmdb and transform
        pbar = tqdm(total=self.len())
        logger.info(f'saving to {os.path.basename(self.root)} dataset')
        pbar.set_description(f'Processing dataset')

        for i, data in enumerate(self.lmdb_dataset):
            filename = osp.join(self.processed_dir, f'data_{i}.pt')
            if osp.exists(filename):
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, filename)
            pbar.update(1)
        pbar.close()

    def len(self):
        return len(self.lmdb_dataset)

    def get(self, idx):
        """Implements the logic to load a single graph."""
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


class ScoreFragmentModelDataModule(pl.LightningDataModule):

    def __init__(self, use_labels=True, train_dataset=None, val_dataset=None,
                 test_dataset=None, batch_size=8, num_workers=8,
                 balance_dataset=False, apply_transform=True,
                 random_seed=None, **kwargs):
        super().__init__()
        self.random_seed = random_seed
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_dataset = balance_dataset
        self.apply_transform = apply_transform
        self.transform = self.get_data_transform(use_labels, kwargs) if apply_transform else None

        # NOTE: For some reasons, torch_geometric.loader.DataLoader will complain
        # when it encounter <class 'str'> when it 3DCNN data mode
        if kwargs['model'] == 'e3nn':
            #self._dataloader_fn = torch_geometric.loader.DataLoader
            self._dataloader_fn = torch_geometric.data.DataLoader
        else:
            self._dataloader_fn = torch.utils.data.DataLoader

    @staticmethod
    def get_data_transform(use_labels, hparams):
        # Transform pdb data into e3nn or cnn3d format
        flag_ligand = not hparams['no_ligand_flag']
        if hparams['model'] == 'e3nn':
            logger.info(f"E3NN data flags: ligand: {flag_ligand}, fragment: {hparams.get('flag_fragment')}")
            transform = E3NN_Transform(use_labels=use_labels,
                                       flag_ligand=flag_ligand,
                                       **hparams,
                                       )
        elif hparams['model'] == 'cnn3d':
            transform = CNN3D_Transform(use_labels=use_labels,
                                        flag_ligand=flag_ligand,
                                        **hparams,
                                        )
            logger.info(f"Grid config: {transform.grid_config}")
        else:
            raise ValueError(f"Invalid model option {hparams['model']:}")
        return transform

    def prepare_data(self):
        """ Only called on 1 GPU/TPU in distributed """
        pass

    def setup(self, stage=None):
        """
        Called on each GPU separately. Called on every process in DDP.
        stage defines if we are at fit or test step
        """
        if stage == 'fit' or stage is None:
            print(f'Train datasets: {self.train_dataset}')
            all_train = [FragmentDataset(p, pre_transform=self.transform) for p in self.train_dataset]
            self.train = ConcatDataset(all_train)
            print(f'Val datasets: {self.val_dataset}')
            all_val = [FragmentDataset(p, pre_transform=self.transform) for p in self.val_dataset]
            self.val = ConcatDataset(all_val)
        if stage == 'test' or stage is None:
            # Evaluate the model on the held out test set
            print(f'Test datasets: {self.test_dataset}')
            all_test = [FragmentDataset(p, pre_transform=self.transform) for p in self.test_dataset]
            self.test = ConcatDataset(all_test)

    def train_dataloader(self):
        """ Returns training dataloader """
        sampler = None
        if self.balance_dataset:
            temp_dataset = ConcatDataset([LMDBDataset(p) for p in self.train_dataset])
            sampler = create_balanced_sampler(temp_dataset)
        return self._dataloader_fn(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            )
        return train

    def val_dataloader(self):
        """ Returns validation dataloader """
        return self._dataloader_fn(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            )

    def test_dataloader(self):
        """ Returns test dataloader """
        return self._dataloader_fn(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            )
