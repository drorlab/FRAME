import os
import torch
import tqdm
import torch_geometric as tg

from src.models.score_intermediate.util.data_transform import *


class E3NN_Transform():
    def __init__(self, use_labels, dataset, element_set, num_nearest_neighbors,
                 use_dummy=False, flag_ligand=True, flag_fragment=False,
                 sample_pocket_atoms_frac=1.0, label_noise_sigma=0.0,**kwargs):
        self.dataset = dataset
        self.flag_ligand = flag_ligand
        self.flag_fragment = flag_fragment

        if dataset == 'per_atom':
            self.data_transform = DataTransform_PerAtom(use_labels, dataset, element_set, use_dummy, flag_ligand,
                         flag_fragment, sample_pocket_atoms_frac, label_noise_sigma)
        elif dataset == 'frag_type':
            self.data_transform = DataTransform_FragType(use_labels, dataset, element_set, use_dummy, flag_ligand,
                                                            flag_fragment, sample_pocket_atoms_frac, label_noise_sigma)
        else:
            self.data_transform = DataTransform(use_labels, dataset, element_set, use_dummy, flag_ligand,
                                                         flag_fragment, sample_pocket_atoms_frac, label_noise_sigma)
        '''
        super().__init__(use_labels, dataset, element_set, use_dummy, flag_ligand,
                         flag_fragment, sample_pocket_atoms_frac, label_noise_sigma)
        '''
        self.num_nearest_neighbors = num_nearest_neighbors

    def __repr__(self):
        return f"e3nn--nn-{self.num_nearest_neighbors},{self.data_transform.__repr__()}"

    __str__ = __repr__

    def get_dims(self):
        return {'in_dim': self.data_transform.num_channels}

    def __call__(self, item):
        # Apply general transformation
        item = self.data_transform(item)

        #convert to torch tensors
        geometry = torch.tensor(item['coords'], dtype=torch.float32)
        features = torch.tensor(item['one_hot'], dtype=torch.float32)
        label = torch.tensor([item['label']], dtype=torch.float32)

        # Figure out the neighbors
        close_top_N = compute_closest_N(geometry, self.num_nearest_neighbors)

        nei_list, geo_list = make_graph_edge_data(geometry, close_top_N)

        data = tg.data.Data(
            x=features,
            edge_index=nei_list,
            edge_attr=geo_list,
            pos=geometry,
            Rs_in=[(self.data_transform.num_channels, 0)],
            label=label,
            id=item['id'],
            file_path=item['file_path'],
            )
        if self.dataset == 'per_atom':
            data['label'] = torch.tensor(data['label']).transpose(1, 0)
            data['select_atoms_index'] = torch.tensor(item['select_atoms_index'])
            data['select_atoms_number'] = torch.tensor(item['select_atoms_number']).transpose(1, 0)
        if 'weight' in item:
            data['weight'] = torch.tensor([item['weight']], dtype=torch.float32)
        elif self.dataset == 'frag_type':
            data['label'] = torch.tensor(data['label']).squeeze(2) #.transpose(1, 0)
        return data


def make_graph_edge_data(geometry, close_top_N):
    nei_list = [] #a list of lists of the edge indices
    geo_list = [] #a vector corresponding to the edge
    for source, x in enumerate(close_top_N.indices):
        #nei_list is a list of all the edges of source
        nei_list.append(
            torch.tensor(
                [[source, dest] for dest in x], dtype=torch.long))

        cart = geometry[x]
        geo_list.append(cart - geometry[source])

    nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
    geo_list = torch.cat(geo_list, dim=0)
    return nei_list, geo_list


def compute_closest_N(geometry, num_nearest_neighbors):
    """

    Parameters
    ----------
    geometry (torch.tensor, nxd where n is number of points and d is dimension)
    num_nearest_neighbors (int)

    Returns
    -------
    (values: torch.tensor of distances to top n, indices: torch.tensor of indices to top n)
    """
    ra = geometry.unsqueeze(0)
    rb = geometry.unsqueeze(1)
    pdist = (ra - rb).norm(dim=2)
    tmp = torch.topk(-pdist, min(num_nearest_neighbors, pdist.shape[0]), axis=1)
    return tmp


if __name__=="__main__":
    from atom3d.datasets import LMDBDataset
    import dotenv as de
    de.load_dotenv(de.find_dotenv(usecwd=True))

    dataset_name = 'per_atom'
    data_dir = 'DATA_DIR' if ((dataset_name in ['fragment', 'per_atom', 'frag_type'])) else 'PDBBIND_DATA_DIR'

    transform = E3NN_Transform(True,
                               dataset_name,
                               'group',
                               num_nearest_neighbors=30,
                               use_dummy=False,
                               flag_ligand=True,
                               flag_fragment=False,
                               #sample_pocket_atoms_frac=0.8,
                               #label_noise_sigma=0.5
                               )

    for mode in ['val', 'test', 'train']:
        dataset_path = os.path.join(os.environ[data_dir], mode)
        print(f"\nCreating dataloaders for {dataset_path:}...")
        dataset = LMDBDataset(dataset_path, transform=transform)
        print(f"Loaded {mode:} dataset with size {len(dataset):}...")

        dataloader = tg.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,#(dataset_name != 'fragment'),
            #sampler=create_balanced_sampler(LMDBDataset(dataset_path)) if (dataset_name == 'fragment') else None,
            )
        print(f"Finished creating {mode:} dataloader of final size {dataloader:}")

        ids = set()
        num_labels = 0
        #non_zeros = []
        batch_sizes = []
        for batch in tqdm.tqdm(dataloader):
            #print(batch)
            batch_labels = batch.label.numpy()
            num_labels += len(batch_labels)
            batch_sizes.append(len(batch_labels))
            #non_zeros.append(np.count_nonzero(batch_labels))
            ids.update(batch.id)
            #import pdb; pdb.set_trace()
        #print(f"Count {sum(non_zeros):}/{num_labels:} non-zero labels...")
        print("Batch sizes:", batch_sizes)
        #print("Non-zeros per batch:", non_zeros)
        print(f"Num of unique ids: {len(ids):}")
