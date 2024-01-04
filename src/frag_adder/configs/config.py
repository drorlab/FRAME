default_config_random = dict(
    organic_fragfiles = 'src/utils/fragment_info/selected_frag_names',
    dataset_name = "dataset_v4_ap",
    adder_type = "random",
    goal_type = "heavy",
    clash_threshold=5, #2.5
    self_clash_threshold=15.0, #13
    hydroxyl_clash_threshold=1.6,
    num_dihedrals=20,
    max_open_bonds=2,
    max_fragments=None,
    max_dihedrals=None,
    max_depth=5,
    debug=False,
    random_seed=10,
    advanced_config=dict(
        debug_output_root="",
        save_scored_candidates=False,
        save_open_bond_candidates=False,
        save_solutions=True,
        log_to_file=True,
        log_file='debug.log'
    ),
    #ML
    batch_size=16
)

default_config_FRAME = dict(
    organic_fragfiles = 'src/utils/fragment_info/selected_frag_names',
    ckpt_path_fragment = "/oak/stanford/groups/rondror/users/lxpowers/ligand_building/training/models_wandb/wandb/run-20230210_023159-ap_fragment_v5_p_pre_4/files/checkpoints/FragS-epoch=030-val_loss=0.3191.ckpt",
    ckpt_path_open_bond = '/oak/stanford/groups/rondror/users/lxpowers/ligand_building/training/models_wandb/wandb/run-20221130_165133-ap_per_atom_2/files/checkpoints/FragS-epoch=049-val_loss=0.0557.ckpt',
    dataset_name = "dataset_v4_ap",
    adder_type = "ML_2model",
    goal_type = "heavy",
    clash_threshold=5, #2.5
    self_clash_threshold=15.0, #13
    hydroxyl_clash_threshold=1.6,
    num_dihedrals=20,
    max_open_bonds=1,
    max_fragments=None,
    max_dihedrals=None,
    max_depth=5,
    debug=False,
    random_seed=10,
    advanced_config=dict(
        debug_output_root="",
        save_scored_candidates=False,
        save_open_bond_candidates=False,
        save_solutions=True,
        log_to_file=True,
        log_file='debug.log'
    ),
    batch_size=16
)

def get_config(name):
    return {
       "config_random": default_config_random,
        "config_ML": default_config_FRAME,
    }[name]