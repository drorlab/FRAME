import torch
import torch_geometric

from src.frag_adder.data_transforms import prepare_model, maestruct_to_df_simple
from src.frag_adder.heuristic_adder import HeuristicFragmentAdder
from src.utils.fragment_info.fragment_files import read_fragment_names_from_file
from src.models.score_intermediate.util.load_ckpt import load_checkpoint
from src.frag_adder.ligand_node import LigandNode_List

class OpenBond(HeuristicFragmentAdder):
    def __init__(self, model, transform, device, config):
        super().__init__(config)
        self.model_bond = prepare_model(model, device)
        self.transform_bond = transform
        self.device = device
        self.batch_size = config['batch_size']

    def open_bond_heuristic(self, current):
        # extract the hydrogens

        ligand_df = maestruct_to_df_simple(current.ligand, atom_label=True)
        item = {
            'atoms_pocket': maestruct_to_df_simple(current.protein, atom_label=True),
            'atoms_ligand': ligand_df,
            'label': None,
            'id': None,
            'file_path': None
        }

        #print(item['atoms_pocket'])
        #print(item['atoms_ligand'])

        if self.transform_bond:
            item = self.transform_bond(item)
        dataloader = torch_geometric.data.DataLoader([item], batch_size=1)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                pred = -self.model_bond(batch)
                #print('pred', pred)
                pred = pred.squeeze()
                ids = batch.select_atoms_index

        ids = ids[0].tolist()
        open_bond_pred = pred.tolist()

        #print('ids', ids)
        #print('pred', open_bond_pred)

        st_ids = ligand_df.iloc[ids]['serial_number']

        #print('st_ids', st_ids)
        self.logger.info(f"\tSmallest open bond score is {min(open_bond_pred)}")
        open_bond_score_dict = {A: B for A, B in zip(st_ids, open_bond_pred)}
        # print(ids, open_bond_score_dict)
        return open_bond_score_dict

    def sample_open_bonds(self, parent_node):
        """
        Scan over bonds in a ligand bound to a protein and return a list of
        candidates of open bonds within the ligand when attached to the hydroxyl
        has steric clash with the protein less than some threshold.
        :param parent_node: LigandNode object
        :type parent_node: LigandNode
        :return: List of pair of bonded atoms where the second atom is hydrogen
        :rtype: [(schrodinger.structure._StructureAtom,
                  schrodinger.structure._StructureAtom)]
        """
        print("sample open bonds")
        open_bonds = self.get_open_bonds(parent_node.ligand)

        self.logger.info(f"\tSampling open bonds from {len(open_bonds):} bonds "
                         f"and total {parent_node.ligand.atom_total:} atoms...")

        if (len(open_bonds) < 2):
            return open_bonds

        for x in open_bonds:
            print("open bond:", x[0].index, x[1].index)

        open_bond_score_dict = self.open_bond_heuristic(parent_node)
        #the second index should be hydrogen so we use that atom index as lookup key
        self.lookup_bond_scores = lambda open_bond: open_bond_score_dict[open_bond[1].index]

        if (self.debug_config['save_open_bond_candidates']):
            root = self.debug_config['debug_output_root']
            file_name = f"{root}d{parent_node.depth + 1}_open_bond_candidates.mae"

            # generate the candidates
            candidates = LigandNode_List()
            for open_bond in open_bonds:
                # Try if the open bond has enough space to add fragment. Use
                # Hydroxide as a proxy to test for clash
                test_node = self.add_fragment_to_node(parent_node, open_bond, 'Hydroxide')
                test_node.score = self.lookup_bond_scores(open_bond)
                test_node.h_atom_id = open_bond[1].index
                candidates.append(test_node)

            candidates.sort_by_score()

            def title_format_func(node):
                hid = node.h_atom_id
                return f'{hid}_{open_bond_score_dict[hid]:.3}'

            candidates.write_to_file(file_name, title_format_func)

        return open_bonds

    def open_bond_scorefxn(self, current, goal):
        return self.lookup_bond_scores(current)

    def fragment_scorefxn(self, current, goal):
        pass

    def dihedral_scorefxn(self, current, goal):
        pass

    def cost(self):
        pass

    def heuristic(self):
        pass

def initialize_1model_adder(config):
    """
    Use E3NN/3DCNN based model trained on PDBBind dataset as heuristic
    :return: the configured greedy fragment adder
    :rtype: GreedyFragmentAdder
    """
    ckpt_path = config['ckpt_path']

    print(f'Load model from {ckpt_path}')
    model, transform = load_checkpoint(ckpt_path)
    print(model)

    organic_fragname_list = read_fragment_names_from_file(config['organic_fragfiles'])
    print(f"Loading {len(organic_fragname_list):} fragments from the library")
    config['fragname_list'] = organic_fragname_list

    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return OpenBond(model, transform, device, config)
