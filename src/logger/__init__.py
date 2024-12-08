from types import Dict

import enum
import wandb

from dataclasses import dataclass

class LoggerMode(enum.Enum):
    TENSOR_BOARD=0
    WANDB=1

class Logger:
    def __init__(self, project_name: str, config: dataclass, mode: LoggerMode):
        if mode == LoggerMode.TENSOR_BOARD:
            raise NotImplementedError
        elif mode == LoggerMode.WANDB:
            self.wandb_run = wandb.init(project=project_name, config=config)
        else:
            raise ValueError("Invalid logger mode")

    def log(self, data: Dict, step: int):
        self.wandb_run.log(data, step=step)

