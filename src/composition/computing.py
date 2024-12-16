from dataclasses import dataclass

import torch


@dataclass
class ComputeConfig:
    # cluster environment
    # distributed config
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __manual_post_init__(self):
        self.device = torch.device(self.device)
