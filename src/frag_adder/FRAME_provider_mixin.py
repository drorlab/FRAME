from abc import ABC, abstractmethod
from schrodinger.structure import Structure

class I_FRAME_ProviderMixin(ABC):

    @abstractmethod
    def get_pocket(self, example_id: str) -> Structure:
        pass

    @abstractmethod
    def get_seed_ligand(self, example_id: str) -> Structure:
        pass

    @abstractmethod
    def get_endpoint_ligand(self, example_id: str) -> Structure:
        pass

    @abstractmethod
    def get_grid(self, example_id: str) -> str:
        pass

