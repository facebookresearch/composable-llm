try:
    from .hawk import LMHawk
    from .mamba import LMMamba
    from .mingru import LMMinGRU
    from .minlstm import LMMinLSTM
    from .rnn_utils import LMFastRNNArgs
except ImportError as e:
    print(e)
    print(
        "Could not import SSM. This is likely due to the lack of installation of the ssm dependencies."
        "You may install them with `pip install .[ssm]."
    )
