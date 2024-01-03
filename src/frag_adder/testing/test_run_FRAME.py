from src.dataset_apis.source.pdb_dataset import PDBDataset
from src.frag_adder.run_FRAME import run_FRAME
from src.frag_adder.configs.config import get_config

class Test_Dataset():
    def __init__(self):
        self.pdb_dataset = PDBDataset('/scratch/groups/rondror/lxpowers/temp_output/pdb_files_v1')

    def get_seed_ligand(self, P):
        return self.pdb_dataset.get_ligand_struc(P)

    def get_pocket(self, P):
        return self.pdb_dataset.get_pocket_struc(P)

    def get_endpoint_ligand(self, P):
        return self.pdb_dataset.get_ligand_struc('3udn_09B')

'''
srun --cpus-per-task=4 --cores-per-socket=4 -t 2:00:00 -p rondror --gres=gpu:1 --constraint=GPU_SKU:TITAN_Xp --pty bash -i -l
$SCHRODINGER/run python3 -m src.frag_adder.testing.test_run_FRAME
'''

if __name__ == '__main__':
    mocked_dataset = Test_Dataset()
    config = get_config("config_ML2")
    ids = ['3udh_091']
    run_FRAME(mocked_dataset, config, ids, 'test_pairs')