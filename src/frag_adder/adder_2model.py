from src.frag_adder.data_transforms import prepare_model, maestruct_to_df_simple, ligand_node_list_to_loader
from src.frag_adder.adder_bond_model import OpenBond

from src.models.score_intermediate.util.load_ckpt import load_checkpoint
from src.utils.fragment_info.fragment_files import read_fragment_names_from_file

import torch
import random
import numpy as np

class TwoModel_Adder(OpenBond):
    def __init__(self, model_bond, transform_bond, model_frag, transform_frag, device, config):
        super().__init__(model_bond, transform_bond, device, config)
        self.model_frag = prepare_model(model_frag, device)
        self.transform_frag = transform_frag

    def fragment_scorefxn(self, current, goal):
        return random.random()

    def dihedral_scorefxn(self, current, goal):
        return random.random()

    def heuristic(self, current_list, goal):
        dataloader = ligand_node_list_to_loader(
            current_list,
            self.transform_frag,
            split_ligand=self.transform_frag.__dict__.get('flag_ligand', False),
            batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                # Need to negate the score since higher predicted score means
                # better affinity
                pred = -self.model_frag(batch)
                preds.extend(pred.tolist())
        return np.array(preds)

    def cost(self, from_node, to_node_list):
        return np.array([0.0 for n in to_node_list])


def initialize_2model_adder(config):
    """
    Use E3NN/3DCNN based model trained on PDBBind dataset as heuristic
    :return: the configured greedy fragment adder
    :rtype: GreedyFragmentAdder
    """
    print('open bond model', config['ckpt_path_open_bond'])
    print('fragment model', config['ckpt_path_fragment'])
    model1, transform1 = load_checkpoint(config['ckpt_path_open_bond'])
    model2, transform2 = load_checkpoint(config['ckpt_path_fragment'])

    organic_fragname_list = read_fragment_names_from_file(config['organic_fragfiles'])
    print(f"Loading {len(organic_fragname_list):} fragments from the library")
    config['fragname_list'] = organic_fragname_list

    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return TwoModel_Adder(model1, transform1, model2, transform2, device, config)