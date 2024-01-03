import logging
import sys

from abc import ABC, abstractmethod

from schrodinger.structutils import build, interactions

from src.utils.fragment_info.simple_fragment_config import fragment_info
from src.frag_adder.ligand_node import LigandNode
from src.utils.struc_tools import get_bonded_indices
from src.utils.build_molecules import Fragment_Attachment_Set


class FragmentAdder(ABC):
    """
    Base class of fragment adder; samples adding fragments to exisiting ligand
    structure bound to a protein pocket.
    """

    def __init__(self, config):
        """
        Constructor.

        :param config: dictionairy of configuration options, see below
        :type clash_threshold: dict

        :param fragname_list: Names of the fragments to sample from
        :type fragname_list: [str]
        :param fragname_list: Names of the fragments to sample from
        :type fragname_list: [str]
        :param fragment_group: The group the fragments in the list belong to,
            e.g. organic
        :type fragment_group: str
        :param debug: If set to True, print all debug messages
        :type debug: Bool
        """
        self.fragname_list = config['fragname_list']
        self._init_logger(config['debug'])
        self.frag_set = Fragment_Attachment_Set()

    def _init_logger(self, debug):
        """
        Initialize logger to stdout
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def add_fragment_to_node(self, node, open_bond, fragname):
        struct, attachment_atom_idx = self.add_fragment(node.ligand, open_bond, fragname)

        new_node = LigandNode(node.protein, struct,
                              parent_ligand_size=node.ligand.atom_total,
                              depth=node.depth + 1,
                              attachment_atom_idx=attachment_atom_idx,
                              fragname=fragname,
                              original_atom_idx=open_bond[0].index
                              )
        return new_node

    def add_fragment(self, ligand, open_bond, fragname):
        """
        Attach fragment to the non-hydrogen atom in the open bond of the
        ligand structure.

        :param ligand: Ligand structure object
        :type ligand: schrodinger.structure.Structure
        :param open_bond: Pair of bonded atoms where the second atom is hydrogen
        :type open_bond: [(schrodinger.structure._StructureAtom,
                           schrodinger.structure._StructureAtom)]
        :param fragname: Fragment name to attach to the bond
        :type fragname: str

        :return: Structure object with the fragment attached and
            new index of the non-hydrogen atom in the open bond
        :rtype: schrodinger.structure.Structure, int
        """
        from_atom, to_atom = open_bond
        # to_atom should be the hydrogen
        assert to_atom.atomic_number == 1

        struct = ligand.copy()

        # renumbered is a dictionary of atom renumbering. Keys are old atom
        # numbers, values are new atom numbers (or None if the atom was deleted,
        # i.e. a hydrogen).
        # First, the atoms in the original ligand get renumbered, then
        # the attached fragments were assigned to the next available indices


        if (fragname[0] == 'R'):
            #handle custom fragment
            frag_copy = self.frag_set.get_fragment(fragname) #select this from a saved dictionairy
            h_id = self.frag_set.get_attachment_id(fragname) #select this from the title
            #print(fragname, h_id)
            id_base = list(frag_copy.atom[h_id].bonded_atoms)[0].index

            renumbered = build.attach_structure(struct, from_atom, to_atom, frag_copy, id_base, h_id)
            if (self.frag_set.is_tricky(fragname, h_id)):
                self.frag_set.realign(ligand, open_bond, struct, renumbered)
        else:
            renumbered = build.attach_fragment(
                st=struct, fromatom=from_atom.index, toatom=to_atom.index,
                fraggroup=fragment_info[fragname]['group'], fragname=fragname)
        # Figure out the new index of the non-hydrogen atom in the open
        # bond being attached to
        attachment_atom_idx = renumbered[from_atom.index]
        assert not renumbered[to_atom.index], "Hydrogen should have been removed"
        return struct, attachment_atom_idx

    @staticmethod
    def get_open_bonds(structure):
        """
        Return list of pair of bonded atoms, of which one is a hydrogen.
        :param structure: Structure object
        :type structure: schrodinger.structure.Structure

        :return: List of pair of bonded atoms where the second atom is hydrogen,
            i.e. (other atom, hydrogen).
        :rtype: [(schrodinger.structure._StructureAtom,
                  schrodinger.structure._StructureAtom)]
        """
        bonds = []
        for bond in structure.bond:
            if bond.atom1.atomic_number == 1:   # Check if atom1 is hydrogen
                bonds.append((bond.atom2, bond.atom1))
            elif bond.atom2.atomic_number == 1: # Check if atom2 is hydrogen
                bonds.append((bond.atom1, bond.atom2))
        return bonds

    @staticmethod
    def get_bond_indices(ligand, atom_index):
        """
        From a ligand, pick out the indices of the atoms that are bound to the
        atom at the given atom index. Sort by the indices.

        :param ligand: Ligand structure object
        :type ligand: schrodinger.structure.Structure
        :param atom_index: Index of the atom of interest (sorted in ascending order)
        :type: int
        """
        return get_bonded_indices(ligand, atom_index)


    @abstractmethod
    def run(self, ligand, protein, output_filename, **kwargs):
        """
        Main function to initiate the fragment adding process.
        This method must be implemented in a subclass.

        :param ligand: Ligand structure object
        :type ligand: schrodinger.structure.Structure

        :param output_filename: Path to output schrodinger file
        :type output_filename: str
        :param kwargs: Other optional arguments relevant to a particular
            FragmentAdder class
        """
        raise NotImplementedError
