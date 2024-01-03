import torch
import math
import random

import numpy as np

from src.models.score_intermediate.util.element_mappings import ELEMENT_MAPPINGS


class DataTransform(object):
    def __init__(self, use_labels, dataset, element_set, use_dummy, flag_ligand,
                 flag_fragment, sample_pocket_atoms_frac, label_noise_sigma):
        self.use_labels = use_labels
        self.dataset = dataset
        self.element_set = element_set
        self.use_dummy = use_dummy
        self.flag_ligand = flag_ligand
        self.flag_fragment = flag_fragment
        self.sample_pocket_atoms_frac = sample_pocket_atoms_frac
        self.label_noise_sigma = label_noise_sigma

        assert self.element_set in ELEMENT_MAPPINGS
        self.element_mapping = ELEMENT_MAPPINGS[self.element_set].copy()
        # Multiple elements may be mapped to the same encoding, so we
        # need to take the max + 1 (assume encoding starts from 0)
        self.num_unique_elements = max(self.element_mapping.values()) + 1

        if self.use_dummy:
            self.element_mapping['_'] = self.num_unique_elements
            self.num_unique_elements += 1

        # Number of channels
        self.num_channels = self.num_unique_elements
        if self.flag_ligand:
            self.num_channels += 1
        if self.flag_fragment:
            self.num_channels += 1

    def __repr__(self):
        return f"element-{self.element_set},dummy-{self.use_dummy},flag_lig-{self.flag_ligand}," \
               f"flag_frag-{self.flag_fragment},sfrac-{self.sample_pocket_atoms_frac},sigma-{self.label_noise_sigma}"

    __str__ = __repr__

    def prep_keys(self, item):
        # create consistent names
        if 'pocket_res' in item:
            item['atoms_pocket'] = item['pocket_res']
            del item['pocket_res']

    def sample_pocket(self, item):
        if self.sample_pocket_atoms_frac < 1.0:
            # Sample frac of atoms from the pocket
            total_num = len(item['atoms_pocket'])
            remove_num = math.ceil((1.0-self.sample_pocket_atoms_frac)*total_num)
            drop_indices = np.random.choice(item['atoms_pocket'].index, remove_num, replace=False)
            item['atoms_pocket'] = item['atoms_pocket'].drop(drop_indices)

    def atoms_no_flag(self, item):
        ### Atom coords, elements, flags, and one-hot encodings
        if self.sample_pocket_atoms_frac == 0:
            return self.atoms_no_flag_no_protein(item)

        pocket_elements = item['atoms_pocket']['element'].to_numpy()
        pocket_coords = item['atoms_pocket'][['x', 'y', 'z']].to_numpy()

        ligand_coords = item['atoms_ligand'][['x', 'y', 'z']].to_numpy()
        ligand_elements = item['atoms_ligand']['element'].to_numpy()

        coords = np.concatenate((ligand_coords, pocket_coords))

        # print('element lists:', ligand_elements, pocket_elements)
        elements = np.concatenate((ligand_elements, pocket_elements))
        # Ligand flags
        ligand_flags = np.expand_dims(np.concatenate((np.ones(ligand_elements.shape[0]),
                                                      np.zeros(pocket_elements.shape[0]))),
                                      axis=1)
        return coords, elements, ligand_flags

    def atoms_no_flag_no_protein(self, item):
        coords =  item['atoms_ligand'][['x', 'y', 'z']].to_numpy()
        elements = item['atoms_ligand']['element'].to_numpy()
        ligand_flags = np.expand_dims(np.ones(elements.shape[0]), axis=1)
        return coords, elements, ligand_flags


    def atoms_frag_flag(self, item):
        ### Atom coords, elements, flags, and one-hot encodings
        pocket_elements = item['atoms_pocket']['element'].to_numpy()
        pocket_coords = item['atoms_pocket'][['x', 'y', 'z']].to_numpy()

        fragment_coords = item['frag_ligand'][['x', 'y', 'z']].to_numpy()
        fragment_elements = item['frag_ligand']['element'].to_numpy()
        core_coords = item['core_ligand'][['x', 'y', 'z']].to_numpy()
        core_elements = item['core_ligand']['element'].to_numpy()

        coords = np.concatenate((core_coords, fragment_coords, pocket_coords))
        elements = np.concatenate((core_elements, fragment_elements, pocket_elements))
        # Ligand and fragment flags
        ligand_flags = np.expand_dims(np.concatenate((np.ones(core_elements.shape[0]),
                                                      np.ones(fragment_elements.shape[0]),
                                                      np.zeros(pocket_elements.shape[0]))),
                                      axis=1)
        frag_flags = np.expand_dims(np.concatenate((np.zeros(core_elements.shape[0]),
                                                    np.ones(fragment_elements.shape[0]),
                                                    np.zeros(pocket_elements.shape[0]))),
                                    axis=1)
        return coords, elements, ligand_flags, frag_flags

    def __call__(self, item):
        #TODO: should be split up by dataset type and into smaller testable functions
        #TODO: should reference the format determined when lmdb is created

        self.prep_keys(item)
        item['label'] = float(item['label'])
        if 'weight' in item:
            item['weight'] = float(item['weight'])

        if self.flag_fragment:
            coords, elements, ligand_flags, frag_flags = self.atoms_frag_flag(item)
        else:
            coords, elements, ligand_flags = self.atoms_no_flag(item)

        # Make one-hot
        one_hot, sel = elements_to_one_hot(self.element_mapping, elements,
                                           num_unique_elements=self.num_unique_elements, use_dummy=self.use_dummy)
        #if not using dummy, remove non group elements
        if not self.use_dummy:
            coords = coords[sel]
            ligand_flags = ligand_flags[sel]
            if self.flag_fragment:
                frag_flags = frag_flags[sel]

        # Add ligand flag to one-hot version of elements
        if self.flag_ligand:
            one_hot = np.concatenate((one_hot, ligand_flags), axis=1)
        if self.flag_fragment:
            one_hot = np.concatenate((one_hot, frag_flags), axis=1)

        # Add coords and one-hot encodings. Ligand first (core + frag), followed
        # by protein related features/coordinates
        item['coords'] = coords
        item['one_hot'] = one_hot

        del item['atoms_pocket']
        return item

class DataTransform_PerAtom(DataTransform):
    def __call__(self, item):

        self.prep_keys(item)
        item['atoms_ligand']['label'] = item['atoms_ligand']['label'].astype(float)
        coords, elements, ligand_flags = self.atoms_no_flag(item)
        one_hot, sel = elements_to_one_hot(self.element_mapping, elements,
                                           num_unique_elements=self.num_unique_elements, use_dummy=self.use_dummy)
        if not self.use_dummy:
            coords = coords[sel]
            ligand_flags = ligand_flags[sel]
            elements = elements[sel]

        prepare_ligand_hydrogen_labels(item, sel, use_pocket=(self.sample_pocket_atoms_frac > 0))
        assert np.all(elements[item['select_atoms_index'][0]] == 'H')
        # Add ligand flag to one-hot version of elements
        if self.flag_ligand:
            one_hot = np.concatenate((one_hot, ligand_flags), axis=1)

        item['coords'] = coords
        item['one_hot'] = one_hot

        del item['atoms_pocket']
        return item

class DataTransform_FragType(DataTransform):

    def __init__(self, use_labels, dataset, element_set, use_dummy, flag_ligand,
                 flag_fragment, sample_pocket_atoms_frac, label_noise_sigma):

        super().__init__(use_labels, dataset, element_set, use_dummy, flag_ligand,
                 flag_fragment, sample_pocket_atoms_frac, label_noise_sigma)
        self.num_unique_classes = 60


    def atoms_bond_flag(self, item):
        pocket_elements = item['atoms_pocket']['element'].to_numpy()
        pocket_coords = item['atoms_pocket'][['x', 'y', 'z']].to_numpy()

        ligand_coords = item['atoms_ligand'][['x', 'y', 'z']].to_numpy()
        ligand_elements = item['atoms_ligand']['element'].to_numpy()

        coords = np.concatenate((ligand_coords, pocket_coords))
        elements = np.concatenate((ligand_elements, pocket_elements))
        # Ligand flags
        ligand_flags = np.expand_dims(np.concatenate((np.ones(ligand_elements.shape[0]),
                                                      np.zeros(pocket_elements.shape[0]))),
                                      axis=1)

        bond_flags = np.zeros(ligand_elements.shape[0])
        bond_flags[item['atoms_ligand']['flag']] = 1
        frag_flags = np.expand_dims(np.concatenate((bond_flags,
                                                      np.zeros(pocket_elements.shape[0]))),
                                      axis=1)
        return coords, elements, ligand_flags, frag_flags

    def __call__(self, item):
        self.prep_keys(item)
        one_hot_label = np.zeros((self.num_unique_classes, 1))
        one_hot_label[int(item['label'])] = 1
        item['label'] = one_hot_label

        if self.flag_fragment:
            coords, elements, ligand_flags, frag_flags = self.atoms_bond_flag(item)
            item['bond_flag'] = frag_flags
            item['ligand_flag'] = ligand_flags
        else:
            coords, elements, ligand_flags = self.atoms_no_flag(item)

        one_hot, sel = elements_to_one_hot(self.element_mapping, elements,
                                           num_unique_elements=self.num_unique_elements, use_dummy=self.use_dummy)

        # if not using dummy, remove non group elements
        if not self.use_dummy:
            coords = coords[sel]
            ligand_flags = ligand_flags[sel]
            if self.flag_fragment:
                frag_flags = frag_flags[sel]

        # Add ligand flag to one-hot version of elements
        if self.flag_ligand:
            one_hot = np.concatenate((one_hot, ligand_flags), axis=1)
        if self.flag_fragment:
            one_hot = np.concatenate((one_hot, frag_flags), axis=1)

        item['coords'] = coords
        item['one_hot'] = one_hot

        del item['atoms_pocket']
        return item

'''

if not self.use_labels:
    # Don't use any label
    item['label'] = 0
else:
    if self.label_noise_sigma > 0.0:
        # Sample gaussian random noise and add it to the label
        if self.dataset == 'per_atom':
            item['atoms_ligand']['label'] = random.gauss(item['atoms_ligand']['label'].astype(float),
                                                         self.label_noise_sigma)
        else:
            item['label'] = random.gauss(item['label'], self.label_noise_sigma)


'''
def calculate_pos_weight(dataset):
    # Assume binary classes and labels are integers (0 or 1)
    labels = [d['label'] for d in dataset]
    classes, class_sample_count = np.unique(labels, return_counts=True)
    print(f"{classes:}: {class_sample_count:}")
    pos_weight = [class_sample_count[0]/(class_sample_count[1] + 1e-5)]
    return torch.as_tensor(pos_weight, dtype=torch.float)


# Weighted sampler for imbalanced classification
def create_balanced_sampler(dataset):
    # Compute samples weight (each sample should get its own weight)
    # Assume labels/classes are integers (0, 1, 2, ...).
    labels = [d['label'] for d in dataset]
    classes, class_sample_count = np.unique(labels, return_counts=True)
    weight = 1. / class_sample_count
    #print(f'The number of weights is {weight}')
    #print(f'The number of labels is {labels}')
    sample_weights = torch.tensor([weight[t] for t in labels])

    #num_samples = int(max(class_sample_count)*2)
    num_samples = len(dataset)
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights,
                                                     num_samples=num_samples,
                                                     replacement=True)
    return sampler

def prepare_ligand_hydrogen_labels(item, sel, use_pocket=True):
    '''

    Parameters
    ----------
    item (dict)
    sel (np array boolean)

    Returns
    -------

    '''
    # We only care about the prediction of the ligand's hydrogens
    ligand_hydrogens = (item['atoms_ligand'].element == 'H')
    item['label'] = item['atoms_ligand'][ligand_hydrogens].label.values  # get the labels for just the ligand hydrogens

    # Calculate the indices of these ligand hydrogens in the full coords array
    if use_pocket:
        pocket_elements = item['atoms_pocket']['element'].to_numpy()
        ligand_hydrogens_flag = np.concatenate((ligand_hydrogens.astype(float).values, np.zeros(pocket_elements.shape[0])))[
            sel]
    else:
        ligand_hydrogens_flag = ligand_hydrogens.astype(float).values[sel]
    item['select_atoms_index'] = np.nonzero(ligand_hydrogens_flag)  # the indexes of the selected atoms
    # the original serial number in the schrodinger structure object
    item['select_atoms_number'] = (item['atoms_ligand'][ligand_hydrogens].serial_number.values,)

def elements_to_one_hot(element_mapping, elements, num_unique_elements=None, use_dummy=True):
    '''

    Parameters
    ----------
    element_mapping (dict of element name to integer id)
    elements (numpy 1D array)
    num_unique_elements (int)
    use_dummy (bool) whether to assign elements not in element_mapping to dummy element, otherwise these elements are removed

    Returns
    -------
    one_hot (numpy array of ints)
    sel (boolean array)

    '''
    if not num_unique_elements:
        num_unique_elements = max(element_mapping.values()) + 1

    if use_dummy:
        # Mask elements not in mapping
        if '_' not in element_mapping:
            element_mapping = element_mapping.copy()
            element_mapping['_'] = num_unique_elements
            num_unique_elements += 1

        elements[np.isin(elements, list(element_mapping.keys()), invert=True)] = '_'
        selected = np.full(elements.shape[0], True)
    else:
        # Filter out elements not in mapping
        selected = np.isin(elements, list(element_mapping.keys()))
        elements = elements[selected]

    elements_int = np.array([element_mapping[e] for e in elements])
    one_hot = np.zeros((elements.shape[0], num_unique_elements))
    one_hot[np.arange(elements.shape[0]), elements_int] = 1
    return one_hot, selected
