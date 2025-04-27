from abc import ABC, abstractmethod

class HoudiniLM(ABC):
    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError("Method `get_name` is not implemented")
    
    @abstractmethod
    def generate(self, original_prompt: str) -> str:
        raise NotImplementedError("Method `generate` is not implemented")
