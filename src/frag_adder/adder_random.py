from src.frag_adder.heuristic_adder import HeuristicFragmentAdder
import random

from src.utils.fragment_info.fragment_files import read_fragment_names_from_file


class Greedy_Random(HeuristicFragmentAdder):
    def __init__(self, config):
        print("Initializing find fragment!")
        print(config)
        super().__init__(config)

    def open_bond_scorefxn(self, current, goal):
        return random.random()

    def fragment_scorefxn(self, current, goal):
        return random.random()

    def dihedral_scorefxn(self, current, goal):
        return random.random()

    def heuristic(self, current, goal):
        random_list = []
        for i in range(0, len(current)):
            random_list.append(random.random())
        return random_list

    def cost(self, from_node, to_node):
        return 0.0



def initialize_random_adder(config):

    organic_fragname_list = read_fragment_names_from_file(config['organic_fragfiles'])
    config['fragname_list'] = organic_fragname_list

    return Greedy_Random(config)