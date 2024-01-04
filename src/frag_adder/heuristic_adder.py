import random
import logging
import numpy as np
import time
import os
from abc import ABC, abstractmethod

from src.frag_adder.base_adder import FragmentAdder
from src.frag_adder.ligand_node import LigandNode, LigandNode_List
from src.utils.fragment_info.simple_fragment_config import fragment_info
from src.utils.struc_tools import num_heavy_atoms

Infinite = float('inf')


class HeuristicFragmentAdder(FragmentAdder, ABC):
    """
    Add fragments sampled according to some rules, heuristics, and search algorithm
    Default is a greedy search but can replace run and search functions
    from a given set to exisiting ligand structure bound to a protein.
    """

    def __init__(self, config):
        """
        Constructor.

        :param config: dictionairy of configuration options, see below
        :type clash_threshold: dict


        :param clash_threshold: Only add fragment to structure if clash is less
            than this threshold
        :type clash_threshold: float
        :param self_clash_threshold: Only add fragment to structure if self-clash
            is less than this threshold
        :type self_clash_threshold: float
        :param hydroxyl_clash_threshold: Clash threshold used for adding hydroxyl
        :type hydroxyl_clash_threshold: float
        :param num_dihedrals: Number of dihedral angles to sample when
            attaching fragment if specified
        :type num_dihedrals: int or None
        param max_open_bonds: Maximum number of open bonds to consider at every
            search state expansion if specified
        :type max_open_bonds: int or None
        param max_fragments: Maximum number of fragments to consider at every
            search state expansion if specified
        :type max_fragments: int or None
        param max_dihedrals: Maximum number of dihedral angles to consider at
            every search state expansion if specified
        :type max_dihedrals: int or None
        param max_depth: Terminate search if the depth exceeds this value
        :type max_depth: int or None
        :param random_seed: Seed to initialize the random number generator.
            If None, the current system time is used (according to python random
            generator default behavior).
        :type random_seed: int
        """
        super().__init__(config)
        self.clash_threshold = config['clash_threshold']
        self.self_clash_threshold = config['self_clash_threshold']
        self.hydroxyl_clash_threshold = config['hydroxyl_clash_threshold']
        self.num_dihedrals = config['num_dihedrals']
        self.max_open_bonds = config['max_open_bonds']
        self.max_fragments = config['max_fragments']
        self.max_dihedrals = config['max_dihedrals']
        self.max_depth = config['max_depth']
        self.debug_config = config['advanced_config']

        if (self.debug_config['log_to_file']):
            fileHandler = logging.FileHandler(self.debug_config['log_file'])
            self.logger.addHandler(fileHandler)

        # self.logger.info(f"Initialize random generator with seed {config['random_seed']:}")
        random.seed(config['random_seed'])

    def run(self, ligand, protein, output_filename, endpoint_struc=None, goal='mw'):

        start = LigandNode(protein, ligand)


        if endpoint_struc != None:
            if goal == 'ref_heavy':
                self.logger.info(f"Using a number heavy atoms of reference as goal")
                goal = 0.95*num_heavy_atoms(endpoint_struc)
                self.goal = {'type': 'heavy', 'value': goal}
            else:
                self.logger.info(f"Using a molecular weight of reference as goal")
                goal = endpoint_struc.total_weight
                self.goal = {'type': 'mw', 'value': goal}
        else:
            self.logger.info(f"Using a number of steps as goal")
            self.goal = {'type': 'depth', 'value': self.max_depth}

        solution = self.greedy_search(start)
        if len(solution) == 1:
            self.logger.info(f"Cannot find any valid fragments to add to ligand...")
            return

        self.logger.info(f"Finished job; write output to {output_filename:}...")

        def title_format_func(node):
            f = node.fragname or "None"
            s = node.score or 0
            return f"d{node.depth}_{f}_{s:.2f}"

        if self.debug_config['save_solutions']:
            solution.write_to_file(output_filename, title_format_func, include_protein=True)

        return solution

    def greedy_search(self, start):
        """
        Greedy search, only picks the best
        :param start: LigandNode instance containing the initial structure
        :type start: LigandNode
        """
        current = start
        solution = LigandNode_List()
        solution.append(start)

        depth = 0

        while (current):
            next_nodes = self.expand_node(current)

            if (len(next_nodes) == 0):
                self.logger.info(f"There are no more options to find because len(next_node) is {len(next_nodes)}")
                break

            heuristic_costs = self.heuristic_timed(next_nodes, self.goal)

            candidates = LigandNode_List()
            for (next, priority) in zip(next_nodes, heuristic_costs):
                candidates.append(next)
                next.score = priority

            candidates.sort_by_score()

            self.logger.info(f"\n\tCandidates founds: {len(candidates):} at depth: {depth:} ")
            self.save_scored_candidates(candidates, depth)

            current = candidates.get(0)
            self.logger.info(f"\tScore of step: {current.score}")
            solution.append(current)
            depth += 1

            solution.write_to_file(os.path.join(self.debug_config['debug_output_root'], 'intermediate.mae'), title_format_func,
                                   include_protein=True)

            if self.is_goal_reached(current, None):
                break

        return solution

    def is_goal_reached(self, current: LigandNode, goal: LigandNode) -> bool:
        """
        Return true when we can consider that 'current' node is the goal
        """
        if self.goal["type"] == "depth":
            return current.depth >= self.goal["value"]
        if self.goal["type"] == "mw":
            mw = current.get_mw()
            goal = self.goal["value"]
            self.logger.info(f"\tCurrent MW is: {mw} Goal MW is: {goal}")
            return mw >= goal
        if self.goal["type"] == "heavy":
            N = current.get_heavy_atom_N()
            goal = self.goal["value"]
            self.logger.info(f"\tCurrent # atoms is: {N} Goal # atoms is: {goal}")
            return N >= goal

    def save_scored_candidates(self, candidates, depth):
        if self.debug_config['save_scored_candidates']:
            self.logger.info(f"\tWriting candidates to file ...")
            root = self.debug_config['debug_output_root']
            file_name = f"{root}d{depth}_final_candidates_by_heuristic.mae"

            def title_format_func(node):
                loc = node.get_attachment_atom()
                return f"d{depth}_{loc}_{node.fragname}_{node.score:.2f}"

            candidates.write_to_file(file_name, title_format_func)

    def expand_node(self, parent_node):
        """
        Given the current protein-ligand state, generate list of valid candidate
        ligand structures with fragment attached. Maximum number of valid
        candidate structures depends on self.max_open_bonds, self.max_fragments,
        and self.max_dihedrals if specified.

        :param parent_node: LigandNode instance containing the current structure
        :type parent_node: LigandNode

        :return: List of valid candidate ligand structures wrapped in
            LigandNode instance
        :rtype: [LigandNode]
        """

        # Generate candidate open bonds
        candidate_open_bonds = self.sample_open_bonds(parent_node)
        candidate_open_bonds = self.filter_top(candidate_open_bonds, self.max_open_bonds, self.open_bond_scorefxn)
        self.logger.info(f"\tFound {len(candidate_open_bonds):} candidate open bonds total")

        # For each candidate open bond, sample few fragments from the library
        candidate_fragments = []
        for open_bond in candidate_open_bonds:
            fragments = self.sample_fragments(parent_node, open_bond, self.fragname_list)
            fragments = self.filter_top(fragments, self.max_fragments, self.fragment_scorefxn)
            candidate_fragments.extend(fragments)
        self.logger.info(f"\tFound {len(candidate_fragments):} candidate molecules total")

        # perform additional reaction filtering
        filtered_candidate_fragments = [n for n in candidate_fragments if self.reaction_filter(n)]
        self.logger.info(f"\t Found {len(filtered_candidate_fragments)} candidate molecules after filtering")

        # for each generated fragment, sample dihedrals
        final_candidates = []
        for new_node in filtered_candidate_fragments:
            candidate_dihedrals = self.sample_dihedrals(new_node, parent_node)
            candidate_dihedrals = self.filter_top(candidate_dihedrals, self.max_dihedrals, self.dihedral_scorefxn)
            self.logger.debug(f"\tFound {len(candidate_dihedrals):} candidate ")
            final_candidates.extend(candidate_dihedrals)

        self.logger.info(f"\tFound {len(final_candidates):} final candidate structures total")
        return final_candidates

    def set_goal(self, goal):
        self.goal = goal

    def is_valid(self, new_node, parent_node):
        """
        Determine the validity of protein-ligand structure.

        :param new_node: The current node under consideration
        :type new_node:  LigandNode
        :param parent_node: The parent of this node
        :type parent_node: LigandNode

        :return: Whether current structure is valid
        :rtype: bool
        """
        original_clash = parent_node.get_protein_clash()
        clash = new_node.get_protein_clash()

        if (clash >= self.clash_threshold + original_clash):
            #print("Fail protein clash", new_node.depth, new_node.fragname, clash-original_clash, self.clash_threshold)
            return False

        self_clash = new_node.get_self_clash()
        if (self_clash > self.self_clash_threshold):
            #print("Fail self clash", new_node.depth, new_node.fragname, self_clash, self.self_clash_threshold)
            return False

        return True

    def sample_dihedrals(self, new_node, parent_node):
        """
        Generate valid candidate structures (i.e. clash is under the threshold,
        etc.) with few dihedral angles which resolution determined by
        self.num_dihedrals.

        :param new_node: The current node under consideration
        :type new_node:  LigandNode
        :param parent_node: The parent of this node
        :type parent_node: LigandNode

        :return: list of Structure objects with new fragment dihedral angle
        :rtype: [LigandNode]
        """
        candidates = []

        if (new_node.fragname in fragment_info):
            symmetry = fragment_info[new_node.fragname]['symmetry']
        else:
            symmetry = 1

        if (not self.num_dihedrals) or (new_node.get_fragment_size() < 1) or (symmetry == 0):
            # if the fragment consists only of 1 atom, no need to adjust for dihedrals.
            # First check the validity of the current angle
            if self.is_valid(new_node, parent_node):
                candidates.append(new_node)

        else:
            # Try a bunch of dihedral angles
            for angle in np.arange(0.0, 360.0 / symmetry, 360.0 / (symmetry * self.num_dihedrals)):
                new_node.adjust_dihedral_angle(angle + 10 * (random.random() - 0.5))
                if self.is_valid(new_node, parent_node):
                    #print(new_node.fragname + ':Y')
                    candidates.append(new_node)
                    new_node = new_node.copy()
                #else:
                #    print(new_node.fragname + ':N')

        return candidates

    def sample_fragments(self, parent_node, open_bond, fragname_list):
        """
        Generate valid candidate structures by attaching fragment from the
        fragment list to the non-hydrogen atom in the open bond.

        :param parent_node: Parent LigandNode
        :type parent_node: LigandNode
        :param open_bond: Pair of bonded atoms where the second atom is hydrogen
        :type open_bond: [(schrodinger.structure._StructureAtom,
                           schrodinger.structure._StructureAtom)]
        :param fragname_list: Names of the fragments to sample from
        :type fragname_list: [str]

        :return: List of Structure objects with the fragment attached including
            the attachment indices and fragment names
        :rtype: [(schrodinger.structure.Structure, int, str)]
        """
        candidates = []
        for fragname in fragname_list:
            new_node = self.add_fragment_to_node(parent_node, open_bond, fragname)
            candidates.append(new_node)

        self.logger.debug(f"\tFound {len(candidates):} candidate fragments for "
                          f"open bond {open_bond[0].index:}-{open_bond[1].index}")

        return candidates

    def sample_open_bonds(self, parent_node):
        """
        Scan over bonds in a ligand bound to a protein and return a list of
        candidates of open bonds within the ligand when attached to the hydroxyl
        has steric clash with the protein less than some threshold.

        :param parent_node: LigandNode object
        :type parent_node: LigandNode

        :return: List of pair of bonded atoms where the second atom is hydrogen
        :rtype: [(schrodinger.structure._StructureAtom,
                  schrodinger.structure._StructureAtom)]
        """
        open_bonds = self.get_open_bonds(parent_node.ligand)

        self.logger.info(f"\tSampling open bonds from {len(open_bonds):} bonds "
                         f"and total {parent_node.ligand.atom_total:} atoms...")

        # calculate the initial clash
        clash_initial = parent_node.get_protein_clash()

        candidates = LigandNode_List()
        filtered_open_bonds = []
        for open_bond in open_bonds:
            # Try if the open bond has enough space to add fragment. Use
            # Hydroxide as a proxy to test for clash
            test_node = self.add_fragment_to_node(parent_node, open_bond, 'Hydroxide')
            clash = test_node.get_protein_clash()
            new_clash = clash - clash_initial
            valid = new_clash < self.hydroxyl_clash_threshold
            candidates.append(test_node)
            if (valid):
                filtered_open_bonds.append(open_bond)

        if (self.debug_config['save_open_bond_candidates']):
            root = self.debug_config['debug_output_root']
            file_name = f"{root}d{parent_node.depth + 1}_open_bond_candidates.mae"

            def title_format_func(node):
                new_clash = node.get_protein_clash() - clash_initial
                result = 'y' if (new_clash < self.hydroxyl_clash_threshold) else 'n'
                return f'{node.get_attachment_atom_original_id()}_{new_clash:.3}_{result}'

            candidates.write_to_file(file_name, title_format_func)

        return filtered_open_bonds

    def filter_top(self, candidates, max_candidates, filter_func):
        if (len(candidates) < 2):
            return candidates
        if max_candidates:
            # Pick the lowest k structures based on some scoring function
            candidates.sort(key=lambda c: filter_func(c, self.goal))
            candidates = candidates[:max_candidates]
        return candidates

    def reaction_filter(self, node):
            atom1 = node.ligand.atom[node.get_attachment_atom()].element
            atom2 = node.ligand.atom[node.get_branchpoint_atom()].element
            hetero = ['O', 'N', 'S', 'F', 'Cl']
            if (atom1 in hetero) and (atom2 in hetero):
                return False
            else:
                return True

    ############################ Search related methods ############################
    @abstractmethod
    def open_bond_scorefxn(self, current: LigandNode, goal: LigandNode) -> float:
        """
        Function to score open bonds (lower is better). Used to select
        max_open_bonds to consider.
        """
        raise NotImplementedError

    @abstractmethod
    def fragment_scorefxn(self, current: LigandNode, goal: LigandNode) -> float:
        """
        Function to score fragments (lower is better). Used to select
        max_fragments to consider.
        """
        raise NotImplementedError

    @abstractmethod
    def dihedral_scorefxn(self, current: LigandNode, goal: LigandNode) -> float:
        """
        Function to score dihedrals (lower is better). Used to select
        max_dihedrals to consider.
        """
        raise NotImplementedError

    def heuristic_timed(self, current_list, goal: LigandNode) -> np.ndarray:
        start_time = time.time()
        heuristic_costs = self.heuristic(current_list, goal)
        end_time = time.time()
        self.logger.info(
            f"\tHeuristic Time elapsed: {end_time - start_time}  Average: {(end_time - start_time) / len(current_list)}")
        return heuristic_costs

    @abstractmethod
    def heuristic(self, current_list, goal: LigandNode) -> np.ndarray:
        """
        Compute the estimated (rough) distance between a list of nodes and the goal.
        current_list is a list of LigandNode objects.
        """
        raise NotImplementedError

    @abstractmethod
    def cost(self, from_node_list: LigandNode, to_node) -> np.ndarray:
        """
        Compute the real cost between two adjacent nodes from_node and to_node.
        to_node_list is a list of nodes.
        """
        raise NotImplementedError

    def reconstruct_path(self, came_from: LigandNode, start: LigandNode,
                         last: LigandNode, include_start=False):
        """
        Return a list of ligand intermediate structures starting from initial
        to final structures.

        :param include_start: If starting ligand should be included in the
            output list
        :type include_start: bool

        :return: List of intermediate ligand structures
        :rtype: [schrodinger.structure.Structure]
        """
        current = last
        path = []
        while current != start:
            print(str(current))
            path.append(current)
            current = came_from[current]
        if include_start:
            path.append(start)
        path.reverse()
        return path

def title_format_func(node):
    f = node.fragname or "None"
    s = node.score or 0
    return f"d{node.depth}_{f}_{s:.2f}"

def add_fragments_random(output_filename, goal, beam_width, num_solutions, **kwargs):
    class FindFragment(HeuristicFragmentAdder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def open_bond_scorefxn(self, current, goal):
            return random.random()

        def fragment_scorefxn(self, current, goal):
            return random.random()

        def dihedral_scorefxn(self, current, goal):
            return random.random()

        def heuristic(self, current_list, goal):
            return np.array([random.random() for n in current_list])

        def cost(self, from_node, to_node_list):
            return np.array([0.0 for n in to_node_list])

        def is_goal_reached(self, current, goal):
            return current.depth >= self.max_depth

    return FindFragment(**kwargs).run(output_filename, goal, beam_width, num_solutions)
