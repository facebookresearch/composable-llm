"""
Vanilla Transformer Training Script.

License
-------
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2024, Meta
"""

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

import gc
import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from timeit import default_timer as timer

from composition.data import DataConfig
from composition.distributed import ComputeConfig
from composition.model import TransformerConfig
from composition.logging import MonitorConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    grad_acc_steps: int = 1
    steps: int = 1000
    pass

# -------------------------------------------------------------------------------
# Configuration Class
# -------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)


def validate_config(config, **kwargs):
    pass

# -------------------------------------------------------------------------------
# Preemption Handling
# -------------------------------------------------------------------------------

@dataclass
class TrainState(Stateful):
    step: int  # nb of steps taken by the optimizer
    acc_step: int  # nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


# global_variables = dict(preemption_flag=False)


# def set_preemption_flag(signum, frame):
#     logger.warning("Signal handler called with signal " + str(signum))
#     logger.warning("Preemption ! checkpointing asap and exiting.")
#     global_variables["preemption_flag"] = True


# -------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------


# def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
#     test = train_state.step % freq == 0
#     if acc_step is not None:
#         test = test and (train_state.acc_step == acc_step)
#     elif acc_freq is not None:
#         test = test and ((train_state.acc_step % acc_freq) == 0)
#     return test


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(config: TrainConfig):

    # # toggle off garbage collection
    # gc.disable()

    # # config checking
    # validate_config(config)

    with ExitStack() as context_stack:

        # # reproducibility
        # # NB TBD: it would make sense to put this in the monitoring context
        # torch.manual_seed(config.monitor.seed)

        # ---------------------------------------------------------------------
        # Build and Parallelize model
        # ---------------------------------------------------------------------

        logger.info("Building model")
        model = Transformer(config.model)

        # Build Optimizer

        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)

        # ---------------------------------------------------------------------
        # Recover Checkpoint
        # ---------------------------------------------------------------------

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=None,
            scheduler=None,
        )

        # ---------------------------------------------------------------------
        # Build Data and Monitoring Context
        # ---------------------------------------------------------------------

        data_loader = context_stack.enter_context(
            DataContext(
                config.data,
                state=train_state.data_loader_state,
            )
        )
        # monitor = context_stack.enter_context(
        #     MonitorContext(
        #         config.monitor, model
        #     )
        # )

        # # garbage collection
        # gc.collect()

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------

        model.train()

        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # -----------------------------------------------------------------
            # Batch of data
            # -----------------------------------------------------------------

            # curr_lr = float(optimizer.param_groups[0]["lr"])
            # data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(
                batch,
                dtype=torch.long,
            )

            # # manual garbage collection otherwise different processes running the GC would slow down the whole pipeline
            # if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
            #     logger.info("garbage collection")
            #     gc.collect()

            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            # data_load_time = round(timer() - data_load_start, 4)
            # nwords_since_last_log += input_ids.numel()

            # bsz, seqlen = labels.shape

            # -----------------------------------------------------------------
            # Forward and backward pass
            # -----------------------------------------------------------------

            preds = model(input_ids)
            loss = loss_func(preds, labels)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / config.optim.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()

            # # For logging we undo that scaling
            # loss = loss.detach() * config.optim.grad_acc_steps

            # optimizer step
            if train_state.acc_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # # updates the scale for next iteration
            # # training iteration complete
            # end_timer.record()
            # torch.cuda.synchronize()
            # curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)


def main():
    """
    The command line interface here uses OmegaConf
    https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments

    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    # Load config from path specified by the `config` cli argument
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # remove 'config' attribute as the underlying config class does not have it
    del cli_args.config

    # Load structured config
    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    # Launch training with the config
    train(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main()
