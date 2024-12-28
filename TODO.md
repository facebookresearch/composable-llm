# TODOS

#### Scaling law with respect to model size

#### Scaling law with respect to the number of data
- [ ] Dataloader: split between train and test. (Probably write to file a train and test set, and read these).
     - Wrap it around the pytorch native dataloader.

#### Additional features
- [ ] Probing.

- [ ] Evaluation : can probing recover the hidden state.
- [ ] Generation: Implement caching mechanisms from meta-lingua.

- [ ] Clean the arguments in manual_post_init. (vocab_size = X.state_dim)

- Script to upload all metrics to wandb. 
Profiler traces to wandb


#### Further improvements
Improvement for `checkpoint.py`
1. Add an eval flag to avoid deleting checkpointing that are going to be used for evaluation, and have not been evaluated yet.

Improvement for `model.py`
1. Benchmark FlexAttention and use it if providing gain.
1. Add a probing mechanism.
1. Implement the caching mechanism.
1. Extrapolate Rope embeddings for sequences much longer than seen at training.
1. Generate completions of prompts of various length at once through funky masking.

Improvement for `train.py`
1. Add gradient clipping.
1. Add mix precision `torch.amp`.

Improvement for `logging`
1. Log the hostname to be able to check defected nodes.
1. Print utilitiy such as amont of memory available for each process.

Improvement for `visualization`
1. script to cast local logging to wandb.
1. notebook to visualize logging with plotly.

Improvement for `profiler`
1. Make the `light` profiler work when pausing and restarting a run (create a ProfilerState).
     - Check if file already exists, and if so, just append to it without writing the header.
1. Add option to log HFU, MFU...


## Commands
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/base.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/sparse.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/dense.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/low_entropy.yaml


#### Debug
python -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml launcher=bash
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml torchrun=True
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml torchrun=True launcher=bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml

#### Personal Debug
python -m src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml launcher=bash
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml torchrun=True
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml torchrun=True launcher=bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml
OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 -m src.apps.gssm.train config=src/apps/gssm/tmp_configs/debug.yaml

## Longer term project

- Model parallelism

Formal math:
xlean
Metagen -> Lean formal ...

- Subsampling / Mamba Hybrid
