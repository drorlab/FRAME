import os
import torch

import numpy as np
import pandas as pd

from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, grid_size

from src.models.score_intermediate.util.element_mappings import ELEMENT_MAPPINGS
from src.models.score_intermediate.util.data_transform import *


class CNN3D_Transform(DataTransform):
    def __init__(self, use_labels, dataset, element_set,
                 random_seed=None, use_dummy=False, flag_ligand=True,
                 flag_fragment=False,
                 sample_pocket_atoms_frac=1.0, label_noise_sigma=0.0,
                 **kwargs):

        super().__init__(use_labels, dataset, element_set, use_dummy, flag_ligand,
                         flag_fragment, sample_pocket_atoms_frac, label_noise_sigma)

        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Radius of the grids to generate, in angstroms.
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        for k in self.grid_config:
            if k in kwargs:
                self.grid_config[k] = kwargs[k]

    def __repr__(self):
        return f"cnn3d--seed-{self.random_seed},rad-{self.grid_config.radius},res-{self.grid_config.resolution}," \
               f"ndirs-{self.grid_config.num_directions},nrolls-{self.grid_config.num_rolls}," \
               f"{super().__repr__()}"

    __str__ = __repr__

    def get_dims(self):
        return {'in_dim': self.num_channels, 'spatial_size': grid_size(self.grid_config)}

    def get_grid(self, item, center, rot_mat=np.eye(3, 3)):
        resolution = self.grid_config.resolution
        # Grid size and radius
        size = grid_size(self.grid_config)
        true_radius = size * resolution / 2.0

        # Center atoms.
        at = item['coords'] - center

        # Apply rotation matrix.
        at = np.dot(at, rot_mat)
        at = (np.around((at + true_radius) / resolution - 0.5)).astype(np.int16)

        # Prune out atoms outside of grid as well as non-existent atoms.
        sel = np.all(at >= 0, axis=1) & np.all(at < size, axis=1)
        at = at[sel]

        # Select valid atoms and form final grid.
        one_hot = item['one_hot'][sel]

        grid = np.zeros((size, size, size, self.num_channels), dtype=np.float32)
        grid[at[:, 0], at[:, 1], at[:, 2]] = one_hot
        return grid

    def voxelize(self, item, rot_mat):
        # Use center of ligand as subgrid center
        ligand_pos = item['atoms_ligand'][['x', 'y', 'z']].astype(np.float32)
        ligand_center = get_center(ligand_pos)
        # Transform protein/ligand into voxel grids and rotate
        grid = self.get_grid(item, ligand_center, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        grid = torch.tensor(grid, dtype=torch.float32)
        return grid

    def __call__(self, item):
        # Apply general transformation
        item = super().__call__(item)

        # Transform protein/ligand into voxel grids.
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        transformed = {
            'x': self.voxelize(item, rot_mat),
            'label': torch.tensor(item['label'], dtype=torch.float32),
            'id': item['id']
            }
        return transformed


if __name__=="__main__":
    from atom3d.datasets import LMDBDataset
    import dotenv as de
    de.load_dotenv(de.find_dotenv(usecwd=True))

    dataset_name = 'pdbbind'
    data_dir = 'DATA_DIR' if (dataset_name == 'fragment') else 'PDBBIND_DATA_DIR'

    dataset_path = os.path.join(os.environ[data_dir], 'val')
    print(f"Loading dataset for {dataset_name:} from {dataset_path:}")
    dataset = LMDBDataset(
        dataset_path,
        transform=CNN3D_Transform(True, dataset_name, 'group', radius=10.0,
                                  use_dummy=True, flag_ligand=True, flag_fragment=False,
                                  sample_pocket_atoms_frac=0.7,
                                  label_noise_sigma=0.2)
        )

    pos_weight = calculate_pos_weight(dataset)
    print(f"Positive weight: {pos_weight:}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=(dataset_name == 'pdbbind'),
        sampler=create_balanced_sampler(LMDBDataset(dataset_path)) if (dataset_name != 'pdbbind') else None,
        )
    print(f"Finished creating dataloader of final size {dataloader:}")

    for batch in dataloader:
        print('feature shape:', batch['x'].shape)
        print('label:', batch['label'])
        break

