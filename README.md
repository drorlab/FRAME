

## FRAME: Fragment-based Molecular Expansion

Code for Geometric deep learning for structure-based ligand design.
A. Powers, H. Yu, P. Suriana, et. al. ACS Central Science, 2023

### Installation

FRAME uses Schrodinger python environment and structure objects (https://newsite.schrodinger.com/python-api/).

Dataset and model weights are available on [Google Drive](https://drive.google.com/drive/folders/1bBjwx8oAETmaEhUQoGoh2ZD7nY1Ld2KJ?usp=sharing).
Download and add files to the `/data/source` folder along with other required files. 

### Running FRAME


Example usage: 
```
$SCHRODINGER/run python3 -m src.frag_adder.run_FRAME
--config_name config_ML 
--output_folder_path ./test_outputs 
--seed_ligand_path ./data/test_inputs/3C49_seed_ligand.mae 
--protein_pocket_path ./data/test_inputs/3C49_pocket.mae 
--end_point number_steps --max_steps 5
```
- seed_ligand_path is a the path to (.mae) file containing the starting ligand to add fragments to.
- protein_pocket_path is a path to (.mae) file containing the protein pocket. The seed_ligand should be aligned or docked to pocket structure.
- Note: recommended to use GPU to run.

Detailed argument descriptions: 
```
--config_name {config_random,config_ML}
                        Most of options for FRAME are specified in configs, see src/frag_adder/configs. Select between random fragments (config_random) or using trained models (config_ML)
--output_folder_path OUTPUT_FOLDER_PATH
                    Folder to output results, will create folder if it does not exist
--seed_ligand_path SEED_LIGAND_PATH
                    The starting ligand .mae file, must be aligned with pocket
--protein_pocket_path PROTEIN_POCKET_PATH
                    The protein pocket .mae file, recommended to select ~5-7 A around ligand
--end_point {number_steps,ref_heavy,ref_mw}
                    Options for when to terminate adding fragments, ref_heavy and ref_mw use provided reference
                    ligand (--endpoint_ligand_path) to determine maximum number of heavy atoms or molecular weight
--max_steps MAX_STEPS
                    If end point is number_steps, maximum number of fragments to add
--endpoint_ligand_path ENDPOINT_LIGAND_PATH
                    If end point is ref_heavy or ref_mw, path to reference .mae file for determine number of
                    fragments to add
--e3nn_env_path 
                    To access e3nn package from SCHRODINGER python, provide path to e3nn libraries
```
