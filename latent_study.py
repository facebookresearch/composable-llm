import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from nanollama.data.gssm import DataConfig, Node, OnlineDataLoader, init_dataloader_state
from nanollama.model import Transformer, TransformerConfig
from nanollama.utils import initialize_nested_object


def read_indented_jsonl(filepath: str) -> list[dict]:
    data = []
    with open(filepath) as file:
        content = file.read()

    # split the content into individual JSON objects
    json_objects = content.split("}\n{")

    # adjust format
    if json_objects:
        json_objects[0] = json_objects[0] + "}"
        json_objects[-1] = "{" + json_objects[-1]
        for i in range(1, len(json_objects) - 1):
            json_objects[i] = "{" + json_objects[i] + "}"

    # parse each JSON object
    for json_str in json_objects:
        json_object = json.loads(json_str)
        data.append(json_object)
    return data


def get_observed_node(code_dir: str, exp: int, grid_id: int) -> Node:
    code_dir = Path(os.path.expandvars(code_dir))
    path = code_dir / "src" / "apps" / "gssm" / "configs" / f"experiment{exp}" / ".gssm_id_path.jsonl"
    with open(path) as f:
        for line in f:
            data_config = json.loads(line)
            if data_config.pop("grid_id") == grid_id:
                break

    all_gssm = read_indented_jsonl(code_dir / f"src/apps/gssm/configs/experiment{exp}/.gssm_id_config.jsonl")
    gssm_config = all_gssm[data_config["gssm_id"]]
    assert gssm_config.pop("gssm_id") == data_config.pop("gssm_id")

    data_config["gssm"] = gssm_config
    data_config["seq_len"] = "FAKE"
    data_config["batch_size"] = "FAKE"
    data_config["asynchronous"] = False
    data_config = initialize_nested_object(DataConfig, data_config)

    state = init_dataloader_state(data_config)
    loader = OnlineDataLoader(data_config, state)
    return loader.node


def get_data(observed_node: Node, bsz: int, seq_len: int) -> dict[str, np.ndarray]:
    all_nodes = {}
    queue = [observed_node]
    while queue:
        node = queue.pop()
        if node.name in all_nodes:
            continue
        all_nodes[node.name] = node
        queue.extend(node.parents)
    data = {key: np.zeros((bsz, seq_len), dtype=int) for key in all_nodes}
    observed_node.initialize(bsz)
    for node in all_nodes:
        data[node][:, 0] = all_nodes[node].state
    for t in range(1, seq_len):
        observed_node.evolve()
        for node in all_nodes:
            data[node][:, t] = all_nodes[node].state
    return data


save_dir = Path(os.path.expandvars("/checkpoint/$USER/icml/logs"))
code_dir = Path(os.path.expandvars("$HOME/code/composable-llm/"))

exp = 103
scaling = "params"
task_id = 10
bsz = 1_000
train_step = 3000
log_dir = save_dir / f"exp{exp}" / scaling

ckpt_path = log_dir / "checkpoints" / str(task_id) / f"{train_step:010}"

# retrieve configuration
config_path = log_dir / "tasks" / f"{task_id}.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
seq_len = config["run_config"]["model"]["block"]["seq_len"] + 1

# retrieve data
grid_id = config["run_config"]["grid_id"]
observed_node = get_observed_node(code_dir, exp, grid_id)
data = get_data(observed_node, bsz, seq_len)


model_config = initialize_nested_object(TransformerConfig, config["run_config"]["model"], inplace=False)

model = Transformer(model_config)
model = torch.compile(model)
model.load_state_dict(torch.load(ckpt_path / "checkpoint.pth")["model"])
