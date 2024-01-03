# E3NN based neural network.

## Installation

See https://github.com/drorlab/ares/blob/main/README.md

## Usage

Copy `.env.template` in `models` to `.env` and update the environment variables to point to the desired paths. The code will read the directory paths
to which it should write the models etc. from this file.

### GPU

To allocate GPU Titan XP node on Sherlock:
`srun --cpus-per-task=4 --cores-per-socket=4 -t 12:00:00 -p rondror --gres=gpu:1 --constraint=GPU_SKU:TITAN_Xp --pty bash -i -l`

**Note that the current Pytorch installation is using CUDA 10.1 which is not supported by the newer GPU node RTX 3090. Make sure to request for Titan XP by specifying the `--constraint` flag**

To train the model on the model on 1 GPU, run the following command:

`DATA=/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/datasets/v3/EXP17_lmdb_dataset/v6/split/data; python train.py $DATA/train $DATA/val -test $DATA/test --gpus=1 --num_workers=8 --batch_size=8 --accumulate_grad_batches=2 --learning_rate=0.00005 --max_epochs=5`
