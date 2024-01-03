import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from src.utils.pysmiles.helpers import bonds_missing, _valence, _bonds
from src.utils.pysmiles.reader import read_smiles
from src.utils.pysmiles.write_smiles import write_smiles
from schrodinger.structure import SmilesStructure
from schrodinger.structutils.analyze import create_nx_graph
from src.utils.struc_tools import get_bonded_hydrogens
from schrodinger.structutils import analyze

def smiles_to_nx_byRDKit(smiles):
	'''
	Convert a smiles format to a networkX graph format using RDkit

	Args:
        smiles (string): the smiles representing molecule

    Returns:
        networkx graph: a molecule graph object
	'''
	test = Chem.MolFromSmiles(smiles)
	return rdkit_mol_to_nx(test) 

def smiles_to_nx_byPYsmiles(smiles, explicit_hydrogens=False):
	return read_smiles(smiles, explicit_hydrogen=explicit_hydrogens)


def rdkit_mol_to_nx(mol):
	'''
	See: https://github.com/maxhodak/keras-molecules/pull/32/files
	'''
	G = nx.Graph()
	for atom in mol.GetAtoms():
			G.add_node(atom.GetIdx(),
									element=atom.GetAtomicNum())
	for bond in mol.GetBonds():
			order = bond.GetBondType()
			if order == Chem.BondType.SINGLE:
					o = 1
			elif order == Chem.BondType.AROMATIC:
					o = 1.5
			elif order == Chem.BondType.DOUBLE:
					o = 2
			elif order == Chem.BondType.TRIPLE:
					o = 3

			G.add_edge(bond.GetBeginAtomIdx(),
									bond.GetEndAtomIdx(),
									order=o)
	return G

FORMAT_VALENCES = {"B": (3,), "C": (4,), "N": (3,), "O": (2,), "P": (3, 5),
            "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,), "H": (1,)}

def nx_to_rdkit_mol(graph):
	m = Chem.MolFromSmiles('')
	mw = Chem.RWMol(m)
	atom_index = {}
	for n, d in graph.nodes(data=True):
		atom = Chem.Atom(d['element'])
		atom_index[n] = mw.AddAtom(atom)
		#mw.GetAtomWithIdx(atom_index[n]).SetFormalCharge(-bonds_missing(graph, n))
		#print(-bonds_missing(graph, n))
		#bonds = _bonds(graph, n, True)
		#bonds += mol.nodes[node_idx].get('hcount', 0)
		#valence = _valence(graph, n, bonds, V=FORMAT_VALENCES)
		#print(d['element'], bonds, valence, bonds-valence)

	for a, b, d in graph.edges(data=True):
		start = atom_index[a]
		end = atom_index[b]
		bond_type = d.get("order")
		if bond_type == 1:
			mw.AddBond(start, end, Chem.BondType.SINGLE)
		elif bond_type == 1.5:
			mw.AddBond(start, end, Chem.BondType.AROMATIC)
		elif bond_type == 2:
			mw.AddBond(start, end, Chem.BondType.DOUBLE)
		elif bond_type == 3:
			mw.AddBond(start, end, Chem.BondType.TRIPLE)
		# more options:
		# http://www.rdkit.org/Python_Docs/rdkit.Chem.rdchem.BondType-class.html
		else:
			mw.AddBond(start, end, Chem.BondType.SINGLE)
			#raise Exception('bond type not implemented')

	mol = mw.GetMol()
	return mol

def nx_to_rdkit_mol_clean(graph):
	return Chem.MolFromSmiles(nx_to_smiles_clean(graph))

def nx_to_rdkit_mol_clean_smart(graph):
	mol, smiles = smiles_to_mol_smart(nx_to_smiles_clean(graph))
	return mol

def nx_to_smiles_clean(graph):
	smiles = write_smiles(clean_graph(graph))
	#smiles = smiles.replace("[NH2]", "[NH2+]")
	return smiles

def clean_graph(G):
    '''
    Remove StereoChemical Information
    '''
    for n, d in G.nodes(data=True):
        if 'stereo' in d:
            d['stereo'] = None
    return G


def smile_to_struc(smile):
	'''
	Convert a smiles format to a schrodinger structure

	Args:
        smile (string): the smiles representing molecule

    Returns:
        networkx graph: a molecule graph object
	'''
	S = SmilesStructure(smile)
	str = S.get3dStructure(require_stereo=False)
	return str

def struc_to_nx(lig_st, aromatic_bond=False, ring_ae_stereo=False):
	'''
	Convert a schrodinger structure to a networkx object

	Args:
        lig_st (schrodinger structure object): the ligand to convert
        aromatic_bond (boolean): whether to convert aromatic bonds to edges with order 1.5
        ring_ae_stereo (boolean): adds an extra stereo attribute to nodes for equitorial vs. axial ring hydrogens

    Returns:
        networkx graph: a molecule graph object
	'''
	if lig_st == None:
		return None
	if aromatic_bond:
		aromatics = set()
		stereo_H = {}
		for ring in lig_st.ring:
			if (ring.isAromatic()):
				for a in ring.atom:
					aromatics.add(a.index)
			else:
				#non_aromatic, label the equitorial vs. axial
				for a in ring.atom:
					if a.element != 'H':
						c = 0
						for h in get_bonded_hydrogens(a):
							stereo_H[h.index] = c
							c = c + 1

				#print('stereo_H', stereo_H)

	lig_graph = create_nx_graph(lig_st)
	for node in list(lig_graph.nodes):
		lig_graph.nodes[node]['element'] = lig_st.atom[node].element
		if ring_ae_stereo:
			if (lig_st.atom[node].element == 'H' and (node in stereo_H.keys())):
				lig_graph.nodes[node]['stereo'] = stereo_H[node]
			else:
				lig_graph.nodes[node]['stereo'] = 0

	for edge in list(lig_graph.edges):
		if aromatic_bond:
			if (edge[0] in aromatics) and (edge[1] in aromatics):
				lig_graph.edges[edge]['order'] = 1.5
			else:
				lig_graph.edges[edge]['order'] = lig_st.getBond(edge[0],edge[1]).order
		else:
			lig_graph.edges[edge]['order'] = lig_st.getBond(edge[0],edge[1]).order
	return lig_graph

def struc_to_smiles(struct):
    return analyze.generate_smiles(struct, unique=True)


def smiles_to_mol_smart(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        while mol is None and '[N]' in smiles:
            smiles = smiles.replace('[N]', '[N+]', 1)
            mol = MolFromSmiles(smiles)
    return mol, smiles
