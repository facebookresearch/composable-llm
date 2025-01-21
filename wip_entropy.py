import json
from itertools import product
from pathlib import Path
from types import TracebackType

import torch
import yaml
from numpy.random import default_rng

from nanollama.data.gssm import DataConfig as GSSMConfig
from nanollama.data.gssm import build_gssm, init_dataloader_state
from nanollama.data.hdf5 import DataConfig, FileEvaluator
from nanollama.utils import initialize_nested_object
from src.apps.gssm.hidden_markov_model import HMM


class EntropyComputer:
    """
    Evaluation manager

    Running evaluation into chunks in order to handle job preemption.

    Usage:
    ```python
    with EvalComputer(*args) as computer:
        while next(computer):
            pass
    """

    def __init__(self, config, path, hmm: HMM) -> None:
        self.data_config = config

        self.path = Path("res.jsonl")
        self.tmp_file = Path("tmp.tmp")
        self.hmm = hmm

        # self.data_config = config.data
        # self.path = Path(config.path) / f"eval_{get_rank()}.jsonl"
        # self.tmp_file = Path(config.path) / f".{get_rank()}_{train_step}.tmp"

        self.step = 0
        self.loss = 0
        self.scaling = 0

    def __enter__(self) -> "EntropyComputer":
        # logger.info("Evaluating model.")

        # retrieve previous computations
        if self.tmp_file.exists():
            with open(self.tmp_file) as f:
                data = json.loads(f)
                self.loss: float = data["loss"]
                self.scaling: float = data["scaling"]
                self.step: int = data["step"]
            # logger.info(f"Found previous evaluation at step {self.step}")
        else:
            self.tmp_file.touch()

        # skip batches that were already evaluated
        self.loader = FileEvaluator(self.data_config).__enter__()
        for _ in range(self.step):
            next(self.loader)

        return self

    @torch.no_grad()
    def __next__(self) -> None:
        try:
            batch, _ = next(self.loader)
            # batch = batch.to(get_local_rank())

            entropy = self.hmm.entropy_of_observations(batch.T).mean().item()

            # evaluate
            self.loss += entropy
            # scaling = batch.size(0) / self.data_config.batch_size
            # self.loss += scaling * loss_func(preds, y_batch).item()
            # self.scaling += scaling
            self.step += 1

            return True

        except StopIteration:
            # rescale loss and save it
            self.loss /= self.scaling
            with open(self.path, "a") as f:
                print(json.dumps({"loss": self.loss, "step": self.train_step}), file=f, flush=True)

            # remove temporary file
            self.tmp_file.unlink()
            return False

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        # if the evaluation was interrupted, save the current state
        if self.tmp_file.exists():
            with open(self.tmp_file, "w") as f:
                print(json.dumps({"loss": self.loss, "scaling": self.scaling, "step": self.step}), file=f, flush=True)

        self.loader.__exit__(exc, value, tb)


if __name__ == "__main__":
    path = "/private/home/vivc/code/composable-llm/src/apps/gssm/configs/experiment1/data.yaml"

    with open(path) as f:
        file_configs = yaml.safe_load(f)

    all_seeds = file_configs["seed"]
    all_nodes = file_configs["gssm"]["nodes"]
    testset_path = file_configs.pop("sets")[1]["path"]
    file_configs.pop("chunk_size")

    all_configs = []  # FAKE

    data_config = {"n_data": 10_000, "batch_size": 100}

    for i, (nodes, seed) in enumerate(product(all_nodes, all_seeds)):
        # specify gssm config
        gssm_config = {}
        gssm_config["gssm"] = file_configs["gssm"]
        gssm_config["gssm"]["nodes"] = nodes
        gssm_config["seed"] = seed
        gssm_config["batch_size"] = "FAKE"
        gssm_config["seq_len"] = "FAKE"

        # retrieve where the generated testset is stored.
        data_config |= {"path": testset_path.replace("$GRIDID", str(i))}

        # write configs and launch the run asynchronously.
        # TODO
        all_configs.append({"data": data_config, "gssm": gssm_config})

    # write configs and launch the run asynchronously.
    for file_config in all_configs:
        # initialize the graph from the config.
        config = initialize_nested_object(GSSMConfig, file_config["gssm"], inplace=False)
        state = init_dataloader_state(config)
        rng = default_rng()
        rng.bit_generator.state = state.graph_rng_state
        node = build_gssm(config.gssm, rng=rng)
        hmm = HMM(top_node=node)

        config = initialize_nested_object(DataConfig, file_config["data"], inplace=False)
        dataloader = FileEvaluator(config)

        # get the data from the hdf5 file specified by the path.
        break

    with EntropyComputer(config, path, hmm) as computer:
        while next(computer):
            print("advancing computation")
