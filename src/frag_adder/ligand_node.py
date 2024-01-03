from schrodinger.structutils import interactions, measure, analyze

from src.utils.molecule_formats import struc_to_smiles
from src.utils.struc_tools import get_bonded_indices, write_mae, num_heavy_atoms


class LigandNode:
    """
    Wrapper for ligand used as a node during heuristic search
    """
    def __init__(self, protein, ligand, parent_ligand_size=0, depth=0,
                 attachment_atom_idx=None,
                 fragname=None,
                 original_atom_idx=None
                 ):
        """
        Constructor.
        :param protein: Protein structure object
        :type protein: schrodinger.structure.Structure
        :param ligand: Ligand structure object associated with this node
        :type ligand: schrodinger.structure.structure
        :param parent_ligand_size: Original size of ligand before the fragment
            is added (i.e. size of the ligand in the parent node). The atom indices
            of the fragment added should start from parent_ligand_size onwards.
        :type parent_ligand_size: int
        :param depth: Number of parents of this node during search
        :type depth: int
        """
        self.protein = protein
        self.ligand = ligand
        self.parent_ligand_size = parent_ligand_size
        self.fragment_size = ligand.atom_total - parent_ligand_size
        self.depth = depth
        self.smiles = None

        self.scores = {}
        self.properties = {}

        #attachment connection information
        self.fragname = fragname
        self.original_atom_idx = original_atom_idx
        self.attachment_atom_idx = attachment_atom_idx
        self.branchpoint_atom_idx = None
        self.dihedral_atoms = None #can be computed later if necessary

        #if necessary, can customize core and frag indices by setting this property
        #userful if indeces do not follow expected numbering
        self.core_indices = None
        self.frag_indices = None

        #calculated properties
        self.protein_clash = None
        self.ligand_clash = None
        self.ligand_weight = self.get_mw()

        #SCORING
        self.score = None
        self.cost = None

    def add_score(self, type, value):
        self.scores[type] = value

    def get_score(self, type):
        if (type in self.scores):
            return self.scores[type]
        else:
            return None

    def add_properties(self, to_add):
        self.properties.update(to_add)

    def get_property(self, type):
        if (type in self.properties):
            return self.properties[type]
        else:
            return None

    def get_properties(self):
        return self.properties

    def get_attachment_atom_original_id(self):
        return self.original_atom_idx

    def get_attachment_atom(self):
        return self.attachment_atom_idx

    def get_branchpoint_atom(self):
        """
        Find the branchpoint if possible (this is the atom that is part of the fragment)
        that replaced hydrogen
        :return: int
        """
        if self.branchpoint_atom_idx is None:

            attachment_atom_bonds = get_bonded_indices(self.ligand, self.get_attachment_atom())
            branchpoint_atom_idx = attachment_atom_bonds[-1]
            assert branchpoint_atom_idx >= self.parent_ligand_size, \
                f"Branch point index {branchpoint_atom_idx:} < {self.parent_ligand_size:}"

            self.branchpoint_atom_idx = branchpoint_atom_idx

        return self.branchpoint_atom_idx

    def get_protein_clash(self):
        if self.protein_clash is None:
            self.protein_clash = interactions.steric_clash.clash_volume(struc1=self.protein, struc2=self.ligand)
        return self.protein_clash

    def get_self_clash(self):
        """
        This calculates how much internal clash there is within a structure.
        """
        if self.ligand_clash is None:
            fragment_indices = self.added_fragment_atom_ids()
            core_indices = self.get_core_indices()

            clash = interactions.steric_clash.clash_volume(self.ligand, core_indices, None, fragment_indices)
            branch_atom = self.get_branchpoint_atom()
            attach_atom = self.get_attachment_atom()
            bond_clash = interactions.steric_clash.clash_volume(self.ligand, [attach_atom], None, [branch_atom])
            self.ligand_clash = clash - bond_clash

        return self.ligand_clash

    def added_fragment_atom_ids(self):
        if self.frag_indices is None:
            return list(range(self.parent_ligand_size, self.ligand.atom_total + 1))
        else:
            return self.frag_indices

    def get_fragment_st(self):
        return self.ligand.extract(self.added_fragment_atom_ids())

    def get_core_indices(self):
        if self.core_indices is None:
            return [a.index for a in self.ligand.atom if (a.index < self.parent_ligand_size)]
        else:
            return self.core_indices

    def get_core_st(self):
        return self.ligand.extract(self.get_core_indices())

    def get_fragment_size(self):
        return self.fragment_size

    def adjust_dihedral_angle(self, angle):
        """
        Adjust the angle of the fragment

        :param angle:
        :return:
        """
        if self.dihedral_atoms is None:
            self.set_dihedral_atoms()

        self.ligand.adjust(angle, atom1=self.dihedral_atoms[0],
                           atom2=self.dihedral_atoms[1],
                           atom3=self.dihedral_atoms[2],
                           atom4=self.dihedral_atoms[3])
        self.protein_clash = None #we don't know these anymore
        self.ligand_clash = None

    def set_dihedral_atoms(self):
        attachment_atom_idx = self.get_attachment_atom()
        branchpoint_atom_idx = self.get_branchpoint_atom()
        attachment_atom_bonds = get_bonded_indices(self.ligand, attachment_atom_idx)
        branchpoint_atom_bonds = get_bonded_indices(self.ligand, branchpoint_atom_idx)
        assert branchpoint_atom_bonds[0] == attachment_atom_idx, \
            f"{branchpoint_atom_bonds:} [0] should be {attachment_atom_idx:}"

        atom1 = attachment_atom_bonds[0]
        atom4 = branchpoint_atom_bonds[-1]

        self.dihedral_atoms = [atom1, attachment_atom_idx, branchpoint_atom_idx, atom4]

    def measure_dihedral(self):
        """
        :return: float angle
        """
        if self.dihedral_atoms is None:
            self.set_dihedral_atoms()
        return measure.measure_dihedral_angle(self.ligand.atom[self.dihedral_atoms[0]],
                                              self.ligand.atom[self.dihedral_atoms[1]],
                                              self.ligand.atom[self.dihedral_atoms[2]],
                                              self.ligand.atom[self.dihedral_atoms[3]])


    def set_core_fragment_indices(self, core_indices, frag_indices):
        self.core_indices = core_indices
        self.frag_indices = frag_indices

    def get_mw(self):
        return self.ligand.total_weight

    def get_heavy_atom_N(self):
        return num_heavy_atoms(self.ligand)

    def copy(self):
        newnode = LigandNode(self.protein,
                          self.ligand.copy(),
                          self.parent_ligand_size,
                          self.depth,
                          self.attachment_atom_idx,
                          self.fragname,
                          self.original_atom_idx)

        newnode.branchpoint_atom_idx = self.branchpoint_atom_idx
        newnode.dihedral_atoms = self.dihedral_atoms

        return newnode

    def title_format_func(self):
        return f"d{self.depth}_{self.fragname}"

    def get_smiles(self):

        if not self.smiles:
            self.smiles = struc_to_smiles(self.ligand)
        return self.smiles

    def __len__(self):
        return self.ligand.atom_total

    def __str__(self):
        return f"{self.ligand} ({self.ligand.atom_total} atoms, " \
               f"depth {self.depth}, fragname {self.fragname}, " \
               f"priority {self.score:.3f}, "

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Assume the protein structure is always the same for all nodes
        return hash((self.ligand))

    def __eq__(self, other):
        if not isinstance(other, LigandNode):
            return False
        ### Assume the protein is the same for all nodes, so need to compare
        # Compare number of atoms
        if self.ligand.atom_total != other.ligand.atom_total:
            return False
        # TODO(psuriana): Compare the SMILES strings
        self_smiles = self.get_smiles() #convert_struct_to_smiles(self.ligand)
        other_smiles = other.get_smiles() #convert_struct_to_smiles(other.ligand)
        if self_smiles != other_smiles:
            return False
        # TODO(psuriana): Compare the 3D conformations
        return True

class LigandNode_List:
    def __init__(self):
        self.nodes = []

    def append(self, node):
        self.nodes.append(node)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, item):
        return self.nodes[item]

    def sort_by_score(self):
        self.nodes = sorted(self.nodes, key=lambda x: x.score)

    def get(self, i):
        return self.nodes[i]

    def write_to_file(self, output_filename, title_format_func, include_protein=False):
        for node in self.nodes:
            node.ligand.title = title_format_func(node)
        strucs = [n.ligand for n in self.nodes]
        if include_protein:
            strucs = [self.nodes[0].protein] + strucs
        write_mae(output_filename, strucs)