from abc import ABC, abstractmethod
from flax import nnx
from flax.struct import dataclass
import optax

class Trainer(ABC):
    @abstractmethod
    def __init__(self, env: "Env", model: nnx.Module, optimizer: "optax.Optim", config: dataclass):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.config = config 

    @abstractmethod
    def make_train(self):
        pass