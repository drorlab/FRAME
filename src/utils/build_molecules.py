from schrodinger.structutils.build import attach_structure
from schrodinger.structutils import transform, rmsd
import os

from src.utils.csv_tools import read_csv
from src.utils.struc_tools import read_mae


class Fragment_Attachment_Set:
	def __init__(self):
		output_root = "/oak/stanford/groups/rondror/users/lxpowers/ligand_building/database_metrics/frag_v1/"
		self.fragment_structures = read_mae(os.path.join(output_root, f'top_{35}.mae'))
		self.fragment_data = read_csv(os.path.join(output_root, f'top_{35}_with_ids.csv'))
		for row in self.fragment_data:
			if (row['tricky'] == ''):
				tricky = []
			else:
				tricky = [int(id) for id in row['tricky'].split(':')]
			row['tricky'] = tricky

	def write_all_names(self, filename):
		names = []
		for i, row in enumerate(self.fragment_data):
			ids = [int(id) for id in row['attach_ids'].split(':')]
			for h_id in ids:
				names.append(f'R{i}:{h_id}')

		with open(filename, 'w') as f:
			for line in names:
				f.write(f"{line}\n")

	def get_fragment(self, name):
		i = int(name.split(':')[0][1:])
		return self.fragment_structures[i]

	def get_attachment_id(self, name):
		h_id = int(name.split(':')[1])  # select this from the title
		return h_id

	def is_tricky(self, name, h_id):
		i = int(name.split(':')[0][1:])
		return h_id in self.fragment_data[i]['tricky']

	def realign(self, parent_struc, open_bond, new_structure, new_ids):
		indices = parent_struc.getAtomIndices()
		indices.remove(open_bond[0].index)
		indices.remove(open_bond[1].index)
		new_indices = [new_ids[i] for i in indices]
		rmsd.superimpose(parent_struc, indices, new_structure, new_indices)






def perform_attachments(target_struc, open_bond, fragment_structures, fragment_data):
	'''
	Find the unique attachment points for fragments in the file (csv)
	Write a new csv file 
	Args:
        target_struc (schrodinger structure): structure to attach fragments to
        open_bond (list of 2 ints): the indices on target_struc to attach fragment
        fragment_structures (list of schrodinger structures): the fragments to attach
        fragment_data (list of dicts): each dictionairy corresponds to fragment_structures and should contain attach_ids, tricky
	'''

	all_struct = []
	for frag, row in zip(fragment_structures, fragment_data):
		ids = [int(id) for id in row['attach_ids'].split(':')]
		if (row['tricky'] == ''):
			tricky = []
		else:
			tricky = [int(id) for id in row['tricky'].split(':')]
		for h_id in ids:
			#copy struct
			target_struc_copy = target_struc.copy()
			frag_copy = frag.copy()
			#find attachment index 
			id_base = list(frag.atom[h_id].bonded_atoms)[0].index
			new_ids = attach_structure(target_struc_copy, open_bond[0], open_bond[1], frag_copy, id_base, h_id)
			
			if (h_id in tricky): #frag_copy.atom[id_base].element == 'N'
				indices = target_struc.getAtomIndices()
				indices.remove(open_bond[0])
				indices.remove(open_bond[1])
				new_indices = [new_ids[i] for i in indices]
				rmsd.superimpose(target_struc, indices, target_struc_copy, new_indices)

			all_struct.append(target_struc_copy)

	return all_struct