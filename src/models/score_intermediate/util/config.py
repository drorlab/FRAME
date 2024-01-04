import os
import dotenv as de
de.load_dotenv(de.find_dotenv(usecwd=True))


DATA_DIR = "/scratch/groups/rondror/lxpowers/temp_output/dataset_v5_ap/lmdb_data/split_seq_30"
defaults = dict(
    ### Training related
    group_name=None,
    project_name='fragment_stitching',
    run_id='train',
    task='binary',
    dataset='fragment',
    train_dataset=[os.path.join(DATA_DIR, 'train')],
    val_dataset=[os.path.join(DATA_DIR, 'val')],
    test_dataset=[os.path.join(DATA_DIR, 'test')],
    batch_size=8,
    num_workers=8,
    random_seed=42,
    early_stopping=False,
    # Data transform related
    element_set='group',
    use_dummy=False,
    no_ligand_flag=False,
    flag_fragment=False,
    sample_pocket_atoms_frac=1.0,
    # Binary task related
    balance_dataset=False,
    upweight_pos=False,
    label_noise_sigma=0.0,
    weighted_loss=False,
    ### Model related
    learning_rate=1e-4,
    lr_schedule='constant',
    pos_threshold=0.5,
    model='e3nn',
    # E3NN model params
    rbf_high=12.0,
    rbf_count=12,
    num_nearest_neighbors=50,
    e3nn_filters=[24, 12],
    fc_filters=[256],
    agg_focus=False, #whether to aggregate only focus atoms for FC layer
    # 3DCNN model params
    conv_drop_rate=0.1,
    fc_drop_rate=0.25,
    batch_norm=False,
    no_dropout=False,
    num_conv=4,
    conv_kernel_size=3,
    fc_units=[512],
    # Pre-trained model params
    pretrained_ckpt_path='',
    freeze_pretrained=False,
    )
