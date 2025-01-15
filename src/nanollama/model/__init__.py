"""
Initialization of the model module

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

from .transfomer import Transformer, TransformerConfig

try:
    from .mamba import Mamba, MambaConfig
except ImportError as e:
    print(e)
    print(
        "Could not import Mamba. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
try:
    from .rnn import FastRNNConfig, Hawk, MinGRU, MinLSTM
except ImportError as e:
    print(e)
    print(
        "Could not import FastRNN. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
except OSError as e:
    print(e)
    print("Could not import FastRNN. This is likely due to missing OS environment variables.")
