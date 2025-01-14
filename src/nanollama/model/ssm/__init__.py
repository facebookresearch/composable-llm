try:
    # from .hawk import LMHawk
    # from .mamba import LMMamba
    from .mingru import MinGRU

    # from .minlstm import LMMinLSTM
    from .rnn_utils import FastRNNConfig
except ImportError as e:
    print(e)
    print(
        "Could not import SSM. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
