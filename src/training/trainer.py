from abc import ABC, abstractmethod
from flax import nnx
from flax.struct import dataclass
import optax

class Trainer(ABC):
    @abstractmethod
    def __init__(self, env: "Env", config: dataclass):
        self.env = env
        self.config = config 

    @abstractmethod
    def make_train(self):
        pass