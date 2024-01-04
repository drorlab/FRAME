from src.models.score_intermediate.enn.model import E3NN_Model, E3NN_Model_Per_Atom, E3NN_Model_Frag_Type
from src.models.score_intermediate.cnn3d.model import CNN3D_Model
import torch
from src.models.score_intermediate.data import ScoreFragmentModelDataModule


def load_checkpoint(filepath):
    checkpoint = torch.load(
        filepath,
        map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']

    ### Load model + weights
    if hparams['model'] == 'e3nn':
        if hparams['dataset'] == 'per_atom':
            model = E3NN_Model_Per_Atom(**hparams)
        elif hparams['dataset'] == 'frag_type':
            model = E3NN_Model_Frag_Type(**hparams)
        else:
            model = E3NN_Model(**hparams)
    elif hparams['model'] == 'cnn3d':
        model = CNN3D_Model(
            dropout=not hparams['no_dropout'],
            **hparams,
            )
    else:
        raise ValueError(f"Invalid model option {hparams['model']:}")

    # Remove "net." from the state_dict keys since it's saved in that format by
    # pytorch_lightning
    state_dict = {k[len('net.'):]: v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    ### Load data transform
    # Data transform
    transform = ScoreFragmentModelDataModule.get_data_transform(False, hparams)
    print(hparams)
    return model, transform
