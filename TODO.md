# TODOS

- [ ] Async creation of the next batch as forward and backward passes are being done.
- [ ] Profiling.
- [ ] Probing.

- [ ] Slurm array option to cross-validate learning rates.

- [ ] Evaluation : can probing recover the hidden state.
- [ ] Generation: Implement caching mechanisms from meta-lingua.

- [ ] Model parallelism

- [ ] Clean the arguments in manual_post_init. (vocab_size = X.state_dim)

- [ ] Dataloader: split between train and test. (Probably write to file a train and test set, and read these).


#### Further improvements
Improvement for `checkpoint.py`
1. Add an eval flag to avoid deleting checkpointing that are going to be used for evaluation, and have not been evaluated yet.

Improvement for `data.gssm.py`
1. Ensure that Z can depend on X.

Improvement for `model.py`
1. Benchmark FlexAttention and use it if providing gain.
1. Add a probing mechanism.
1. Implement the caching mechanism.
1. Extrapolate Rope embeddings for sequences much longer than seen at training.
1. Generate completions of prompts of various length at once through funky masking.

Improvement for `train.py`
1. Add gradient clipping.

Improvement for `logging`
1. Log the hostname to be able to check defected nodes.
1. Print utilitiy such as amont of memory available for each process.

Improvement for `visualization`
1. script to cast local logging to wandb.
1. notebook to visualize logging with plotly.


## Commands
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/base.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/sparse.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/dense.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/low_entropy.yaml


#### Debug
python -m src.apps.gssm.train config=src/apps/gssm/debug_config.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/debug_config.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/debug_config.yaml launcher=bash
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/debug_config.yaml torchrun=True
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/debug_config.yaml torchrun=True launcher=bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train config=src/apps/gssm/debug_config.yaml
OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 -m src.apps.gssm.train config=src/apps/gssm/debug_config.yaml

## Longer term project

Formal math:
xlean
Metagen -> Lean formal ...


