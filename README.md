

## FRAME: Fragment-based Molecular Expansion

Code for Geometric deep learning for structure-based ligand design.
A. Powers, H. Yu, P. Suriana, et. al. ACS Central Science, 2023

### Installation

FRAME uses Schrodinger python environment and structure objects to add fragments (https://newsite.schrodinger.com/python-api/).

For model training, we use pytorch and e3nn library. Install appropriate versions of torch and attendant libraries. Please set the adequate version of CUDA for your system. The instructions shown are for CUDA 11.7. If you want to install the CPU-only version, use CUDA="".
```
TORCH="1.13.0"
CUDA="cu117"
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install pytorch-lightning python-dotenv wandb
```
In addition, we need to install a FRAME-compatible version of the e3nn library (https://github.com/e3nn/e3nn). Please note that this specific version is only provided for compatability, further development should be done using the main e3nn branch.
```
pip install git+ssh://git@github.com/drorlab/e3nn_edn.git
```
We use the LMDB data format from Atom3D (https://www.atom3d.ai/) for fast random access. To install the atom3d package:
```
pip install atom3d
```
Other installation notes: 
- E3nn needs to be built on a version of python that matches the version of the Schrödinger python installation, so that these can be used together (typically Python3.8). 
- Possible installation issues will be posted in `installation_FAQ.md`

Dataset and model weights are available on [Google Drive](https://drive.google.com/drive/folders/1bBjwx8oAETmaEhUQoGoh2ZD7nY1Ld2KJ?usp=sharing).
Download and add files to `/data/source`. 

### Running FRAME


Example usage: 
```
$SCHRODINGER/run python3 -m src.frag_adder.run_FRAME
--config_name config_ML 
--output_folder_path ./test_outputs 
--seed_ligand_path ./data/test_inputs/3C49_seed_ligand.mae 
--protein_pocket_path ./data/test_inputs/3C49_pocket.mae 
--end_point number_steps --max_steps 5
--e3nn_env_path PATH
```
- seed_ligand_path is a the path to (.mae) file containing the starting ligand to add fragments to.
- protein_pocket_path is a path to (.mae) file containing the protein pocket. The seed_ligand should be aligned or docked to pocket structure.
- e3nn_env_path needs to be specified to allow access to e3nn package from SCHRODINGER environment
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
                    To access e3nn package from SCHRODINGER python environment, provide path to e3nn library
```
