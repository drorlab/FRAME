from schrodinger.structure import SmilesStructure, StructureReader, StructureWriter
from schrodinger.structutils import transform
from schrodinger.structutils import build
from schrodinger.structutils import minimize
import time
from schrodinger.forcefield import minimizer

def extract_first_residue_match(st, res_name):
	for residue in st.residue:
		if residue.pdbres == res_name:
			ids = residue.getAtomIndices()
			return residue.extractStructure(), ids

def translate_struc(st, x=0, y=0, z=0):
	transform.translate_structure(st, x, y, z)

def rotate_struc(st, x_angle=0, y_angle=0, z_angle=0, rot_center=None):
	transform.rotate_structure(st, x_angle, y_angle, z_angle, rot_center)

def rotate_struc_centroid(st, x_angle=0, y_angle=0, z_angle=0):
	center = get_centroid(st, None)
	rotate_struc(st, x_angle, y_angle, z_angle, rot_center=list(center))

def read_mae(path):
	return list(StructureReader(path))

def write_mae(path, structures):
	with StructureWriter(path) as stwr:
		for structure in structures:
			stwr.append(structure)
	
def center_struc(st):
	transform.translate_center_to_origin(st)
	return st

def get_centroid(st, atom_list):
	return transform.get_centroid(st, atom_list)[0:3]

def minimize_struc2(conformers, receptor):
	for index, st in enumerate(conformers):
		lig_N = len(st.atom)
		st.extend(receptor)
		minimized = minimizer.minimize_structure(st)

	return [minimized], st

def minimize_struc(conformers, receptor, min):
	minimized = []
	rec_N = len(receptor.atom)
	for index, st in enumerate(conformers):
		lig_N = len(st.atom)
		st.extend(receptor)
		if index == 0:
			receptor_atoms = range(lig_N+1, lig_N+rec_N+1)
			min.deleteAllRestraints()
			min.setStructure(st)
			#min = minimize.Minimizer(struct=st, verbose=True)

			lig_N = len(ligand.atom)
			rec_N = len(receptor.atom)
			ligand.extend(receptor)
			min = minimize.Minimizer(struct=ligand, verbose=True)
			receptor_atoms = range(lig_N + 1, lig_N + rec_N + 1)
			[min.addPosFrozen(i) for i in receptor_atoms]
			min.minimize()
			lig_min = st.extract(range(1, lig_N + 1))

		else:
			min.updateCoordinates(st)

		start_time = time.time()
		min.minimize()
		end_time = time.time()
		print(f"\tMinimize Time elapsed: {end_time - start_time}")

		st = min.getStructure()
		lig_min = st.extract(range(1, lig_N+1))
		minimized.append(lig_min)
	return minimized, st


def get_bonded_hydrogens(atom):
	'''
	Find all hydrogens bonded to atom
	Args:
		atom (schrodinger atom):
	Returns:
		(list of schrodinger atoms)
	'''
	return [a for a in atom.bonded_atoms if a.element == 'H']

def num_heavy_atoms(struc):
	return len([a for a in struc.atom if a.element != 'H'])

def get_Hs(struc):
	return [a for a in struc.atom if a.element == 'H']


def get_heavy(struc):
	return [a for a in struc.atom if a.element != 'H']


def get_bonded_indices(st, index):
	"""
	From a ligand, pick out the indices of the atoms that are bound to the
	atom at the given atom index. Sort by the indices.

	:param ligand: Ligand structure object
	:type ligand: schrodinger.structure.Structure
	:param atom_index: Index of the atom of interest (sorted in ascending order)
	:type: int
	"""
	return sorted([a.index for a in st.atom[index].bonded_atoms])


def get_bonded_elements(st, index):
	return [a.element for a in st.atom[index].bonded_atoms]


def remove_fragment(st, ref_st, heavy_atom_ids):
	"""

	Parameters
	----------
	st (schrodinger structure): the structure to remove fragment from
	ref_st (schrodinger structure): a copy of the structure
	heavy_atom_ids (list of ints): the ids of fragment atoms to remove

	Returns
	-------
	(dict of new ids): Keys are atom numbers before deleting,
	and value for each is the new atom number,or None if that atom was deleted.

	"""

	# get the ids of the attached hydrogens
	hydrogens = []
	branchpoint = None
	for i in heavy_atom_ids:
		for atom_j in st.atom[i].bonded_atoms:
			if (atom_j.element == 'H'):
				hydrogens.append(atom_j.index)
			else:
				if (atom_j.index not in heavy_atom_ids):
					branchpoint = atom_j.index

	indices = heavy_atom_ids + hydrogens

	# delete the heavy atoms and attached hydrogens
	renumber_map = st.deleteAtoms(indices, renumber_map=True)

	# add the hydrogens back to the branchpoint atom
	# print('branchpoint', renumber_map[branchpoint])
	build.add_hydrogens(st, atom_list=[renumber_map[branchpoint]])

	st.title = '{}_{}'.format(renumber_map[branchpoint], len(st.atom))

	correct_dihedrals(st, ref_st, branchpoint, renumber_map, heavy_atom_ids)

	return renumber_map


def correct_dihedrals(st, ref_st, branchpoint, renumber_map, heavy_atom_ids):
	# Need to fix position of certain added hydrogens -OH and -SH
	# is branchpoint an oxygen or sulfur?
	new_branchpoint = renumber_map[branchpoint]
	if (st.atom[new_branchpoint].element in ['O', 'S']):
		# if the branchpoint has 2 bonds, is one hydrogen?
		bonded_elements = get_bonded_elements(st, new_branchpoint)
		if ((len(bonded_elements) == 2) and ('H' in bonded_elements)):

			# compute the previous dihedral from the reference
			# there should be two atoms bonded to original
			bonded_to_branchpoint_original = get_bonded_indices(ref_st, branchpoint)
			# atom2 is not in the removed fragment
			atom2 = [a for a in bonded_to_branchpoint_original if a not in heavy_atom_ids][0]
			# atom4 is the connection atom in the removed fragment
			atom4 = [a for a in bonded_to_branchpoint_original if a in heavy_atom_ids][0]
			# atom1 is connected to atom2, and not branchpoint
			bonded_to_atom2 = get_bonded_indices(ref_st, atom2)
			if len(bonded_to_atom2) > 1:
				atom1 = [a for a in bonded_to_atom2 if a != branchpoint][0]
				old_angle = ref_st.measure(atom1, atom2, atom3=branchpoint, atom4=atom4)

				# compute the new dihedral
				h_added = [a.index for a in st.atom[new_branchpoint].bonded_atoms if a.element == 'H'][0]
				new_angle = st.measure(renumber_map[atom1], renumber_map[atom2],
									   atom3=new_branchpoint, atom4=h_added)

				st.adjust(old_angle,
						  atom1=renumber_map[atom1],
						  atom2=renumber_map[atom2],
						  atom3=new_branchpoint,
						  atom4=h_added)

			# print('dihedrals old:{} new:{}'.format(old_angle, new_angle))


def get_ligand_seed(traj, cutoff=None):
    if (cutoff == None):
        return traj[-1]
    initial_n = num_heavy_atoms(traj[0])
    for i, s in enumerate(traj):
        if (num_heavy_atoms(s)/initial_n < cutoff):
            return s
    return s

def location_key(x,y,z):
	return "{:.2f}-{:.2f}-{:.2f}".format(x,y,z)

def find_corresponding_atom(atoms_1, st_2):
    '''
    both are schrodinger structures
    size of st_extended must be the same or greater to st_base
    find which hydrogens in st_base are missing in st_extended
    '''
    #print('I am in find replaced hydrogens!! \n ##### \n ')
    base_keys = [location_key(a.x, a.y, a.z) for a in atoms_1]
    st2_map = {location_key(a.x, a.y, a.z):a.index for a in st_2.atom if a.element != 'H'}
    replaced_hs = [st2_map[k] for k in base_keys]
    #print(replaced_hs)
    return replaced_hs