import torch_geometric

import collections as col
import pandas as pd
import time

def maestruct_to_df_simple(st, atom_label=False):
    df = col.defaultdict(list)
    for i, a in enumerate(st.atom):
        df['x'].append(a.x)
        df['y'].append(a.y)
        df['z'].append(a.z)
        df['element'].append(a.element.upper())
        df['serial_number'].append(a.index) # Atom index starts from 1
        if atom_label:
            df['label'].append(False)
    df = pd.DataFrame(df)
    # Make up atom names
    return df

def prepare_model(model, device):
    # Set model to test mode
    model.eval()
    # Transfer model to device
    return model.to(device)

def ligand_node_list_to_loader(node_list, transform, split_ligand,
                               batch_size=8):
    """
    Convert list of LigandNode objects into torch_geometric dataloader.
    :param node_list: list of LigandNode objects to convert
    :type node_list: List[LigandNode]
    :param transform: Transform function to apply
    :type transform: DataTransform or function
    :param split_ligand: If set to True, also split the ligand into core and
        fragment ligand coordinates
    :type split_ligand: bool
    :return: dataloader
    :rtype: torch_geometric.data.DataLoader
    """
    ## TODO(psuriana): should we cache the protein pocket? Note that every time,
    # we might get slightly different pocket since we keep adding fragments to
    # the ligand.

    start = time.time()

    items = []
    pocket = maestruct_to_df_simple(node_list[0].protein)

    core_atoms_df = maestruct_to_df_simple(node_list[0].get_core_st())
    print('splitting?', split_ligand)
    for node in node_list:
        # Convert into feature item
        item = {
            'atoms_pocket': pocket,
            'atoms_ligand': maestruct_to_df_simple(node.ligand),
            'label': 0,
            'id': '',
            'file_path': 'search',
        }
        if split_ligand:
            item['core_ligand'] = core_atoms_df
            #this function is used now in case fragments cannot be selected using indices
            item['frag_ligand'] = maestruct_to_df_simple(node.get_fragment_st())
            #print(item['frag_ligand'])
        if transform:
            item = transform(item)
        items.append(item)

    end = time.time()
    print(f"Transform time {end - start} Average: {(end - start) / len(node_list)}")

    loader = torch_geometric.data.DataLoader(items, batch_size=batch_size)
    return loader