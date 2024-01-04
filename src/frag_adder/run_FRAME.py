import sys
import os
import argparse

from src.frag_adder.adder_random import initialize_random_adder
from src.frag_adder.configs.config import get_config

from src.utils.config_tools import write_config
from src.utils.struc_tools import write_mae, read_mae


def run_FRAME(args, config):

    #set up output folder
    output_folder = args.output_folder_path
    config["output_root_folder"] = output_folder
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, 'steps.mae')
    if os.path.exists(output_filename):
        print('Output file already exists, terminating')
        return

    #Setup logging
    config['advanced_config']['debug_output_root'] = output_folder
    config['advanced_config']['log_file'] = os.path.join(output_folder, 'experiment.log')
    write_config(config, os.path.join(output_folder, 'experiment.log'))

    #Setup other configuration options
    adder_type = config["adder_type"]
    config["goal_type"] = args.end_point
    config['max_depth'] = args.max_steps

    #Load input files
    try:
        seed = read_mae(args.seed_ligand_path)[0]
        pocket = read_mae(args.protein_pocket_path)[0]
    except:
        print("Problem loading input files, terminating")
        return

    if args.end_point in ['ref_heavy', 'ref_mw']:
        end_point_ligand = read_mae(args.endpoint_ligand_path)[0]
    else:
        end_point_ligand = None

    if adder_type == "ML_2model":
        ##TODO remove this in final version
        sys.path.insert(0,
                        '/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/envs/e3nn/lib/python3.8/site-packages')
        from src.frag_adder.adder_2model import initialize_2model_adder
        adder = initialize_2model_adder(config)
    if adder_type == 'random':
        adder = initialize_random_adder(config)

    solution = adder.run(seed, pocket, output_filename, endpoint_struc=end_point_ligand, goal=config["goal_type"])
    adder.logger.handlers.clear()

def get_args():
    parser = argparse.ArgumentParser(description='Job Runner')

    parser.add_argument('--config_name', choices=['config_random', 'config_ML'], type=str, default='config_random', help='Most of options for FRAME are specified in configs, see src/frag_adder/configs')
    parser.add_argument('--output_folder_path', type=str, help='Folder to output results, will create folder if it does not exist')
    parser.add_argument('--seed_ligand_path', type=str, help='The starting ligand .mae file, must be aligned with pocket')
    parser.add_argument('--protein_pocket_path', type=str, help='The protein pocket .mae file, recommended to select ~5-7 A around ligand')

    parser.add_argument('--end_point', choices=['number_steps', 'ref_heavy', 'ref_mw'], default='number_steps',
                        help='Options for when to terminate adding fragments, ref_heavy and ref_mw use provided reference ligand (--endpoint_ligand_path) to determine maximum number of heavy atoms or molecular weight')
    parser.add_argument('--max_steps', type=int, default=5, help='If end point is number_steps, maximum number of fragments to add')
    parser.add_argument('--endpoint_ligand_path', type=str, default='', help='If end point is ref_heavy or ref_mw, path to reference .mae file for determine number of fragments to add')

    args = parser.parse_args()
    return args

'''
python3 -m src.frag_adder.run_FRAME --config_name config_random --output_folder_path ./test_outputs 
--seed_ligand_path ./data/test_inputs/3C49_seed_ligand.mae --protein_pocket_path ./data/test_inputs/3C49_pocket.mae --end_point number_steps --max_steps 5
'''
def FRAME_CLI():
    command_line_args = get_args()
    config = get_config(command_line_args.config_name)
    run_FRAME(command_line_args, config)
