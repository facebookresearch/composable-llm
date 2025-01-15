"""
RNN utilities

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from dataclasses import dataclass, field

from .blocklm import BlockLanguageModel, BlockLanguageModelConfig
from .ssm.hawk import HawkBlock
from .ssm.mingru import GRUBlock
from .ssm.minlstm import LSTMBlock
from .ssm.utils_rnn import RNNBlockConfig

# ------------------------------------------------------------------------------
# Configuration class (see ssm.rnn_utils.RNNBlockConfig)
# ------------------------------------------------------------------------------


@dataclass
class FastRNNConfig(BlockLanguageModelConfig):
    implementation: str = None
    block: RNNBlockConfig = field(default_factory=RNNBlockConfig)

    def __post_init__(self):
        super().__post_init__()

        # check validity
        assert self.implementation, "implementation should be specified"
        self.implementation = self.implementation.lower()
        assert self.implementation in ["minlstm", "mingru", "hawk"], f"{self.implementation} not found"

        # inherit parameters from the block model configuration
        for attr in ["emb_dim"]:
            setattr(self.block, attr, getattr(self, attr))

        # default scaling of hidden dimensions
        if self.implementation == "hawk":
            if not self.block.hidden_dim:
                self.block.hidden_dim = 4 * self.emb_dim
            if not self.block.ffn_dim:
                self.block.ffn_dim = 4 * self.emb_dim
        else:
            if not self.block.hidden_dim:
                self.block.hidden_dim = 3 * self.emb_dim


# ------------------------------------------------------------------------------
# Various Architectures
# ------------------------------------------------------------------------------


class Hawk(BlockLanguageModel):
    def __init__(self, config: FastRNNConfig) -> None:
        super().__init__(config, block=HawkBlock)


class MinGRU(BlockLanguageModel):
    def __init__(self, config: FastRNNConfig) -> None:
        super().__init__(config, block=GRUBlock)


class MinLSTM(BlockLanguageModel):
    def __init__(self, config: FastRNNConfig) -> None:
        super().__init__(config, block=LSTMBlock)
