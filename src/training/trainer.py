from abc import ABC, abstractmethod
from flax.struct import dataclass

class Trainer(ABC):
    @abstractmethod
    def __init__(self, env: "Env", config: dataclass):
        self.env = env
        self.config = config 

    @abstractmethod
    def make_train(self):
        pass