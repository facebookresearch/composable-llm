try:
    from .hawk import Hawk

    # from .mamba import LMMamba
    from .mingru import MinGRU
    from .minlstm import MinLSTM
    from .rnn_utils import FastRNNConfig
except ImportError as e:
    print(e)
    print(
        "Could not import SSM. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
