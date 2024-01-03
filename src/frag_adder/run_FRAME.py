import sys
import os
import argparse

from src.frag_adder.FRAME_provider_mixin import I_FRAME_ProviderMixin
from src.frag_adder.adder_random import initialize_random_adder

from src.utils.config_tools import write_config
from src.utils.struc_tools import write_mae

def run_FRAME(dataset: I_FRAME_ProviderMixin, config: dict, ids: list, run_name: str):

    output_folder_root = config["output_root_folder"]
    adder_type = config["adder_type"]
    goal_type = config["goal_type"]

    for P in ids:
        print(P)
        output_folder = os.path.join(output_folder_root, run_name, P)
        config['advanced_config']['debug_output_root'] = output_folder
        config['advanced_config']['log_file'] = f'{output_folder}experiment.log'
        config['goal'] = goal_type

        output_filename = os.path.join(output_folder, 'steps.mae')
        if os.path.exists(output_filename):
            print('exists')
            continue

        os.makedirs(output_folder, exist_ok=True)
        write_config(config, f'{output_folder}config.json')

        seed = dataset.get_seed_ligand(P)
        pocket = dataset.get_pocket(P)
        ligand = dataset.get_endpoint_ligand(P)

        if ligand is None or seed is None or pocket is None:
            print(f'{P} failed')
            print(ligand, seed, pocket)
            continue
        
        if ('debug' in run_name):
            write_mae(f'{output_folder}native.mae', [pocket, ligand])

        if adder_type == "ML_2model":
            ##TODO remove this in final version
            sys.path.insert(0,
                            '/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/envs/e3nn/lib/python3.8/site-packages')
            from src.frag_adder.adder_2model import initialize_2model_adder
            adder = initialize_2model_adder(config)
        if adder_type == 'random':
            adder = initialize_random_adder(config)


        solution = adder.run(seed, pocket, output_filename, endpoint_struc=ligand, goal=goal_type)
        adder.logger.handlers.clear()

def get_args():
    parser = argparse.ArgumentParser(description='Job Runner')
    parser.add_argument('task', choices=['submit_jobs', 'run', 'check'], help='Task to perform')

    parser.add_argument('--start_index', type=int, default=0, help='Starting index')
    parser.add_argument('--total', type=int, default=100, help='Total number of tasks for the submit command')
    parser.add_argument('--number_per_job', type=int, default=1, help='Number of tasks per iteration')

    parser.add_argument('--time', type=str, default="04:00:00", help='Number of tasks per iteration')
    parser.add_argument('--partition', type=str, default='rondror', help='Run partition')
    parser.add_argument('--dry_run', action='store_true', help='Whether to do a dry run')
    parser.add_argument('--submit_all', action='store_true', help='Submit all the jobs, instead of only missing')
    args = parser.parse_args()
    return args

def FRAME_CLI(submit_run, get_ids, output_root, file):
    args = get_args()
