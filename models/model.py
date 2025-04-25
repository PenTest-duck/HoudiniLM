from abc import ABC, abstractmethod

class HoudiniLM(ABC):
    @abstractmethod
    def generate(self, original_prompt: str) -> str:
        raise NotImplementedError("Method `generate` is not implemented")
