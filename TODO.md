# TODOS

- [ ] Profiling: Make sure to understand if we are IO, GPU and CPU bound
     - Switch from html to perfetto for the MemoryProfiler.
     - Visualize Memory footprint with TensorBoard.
     - Visualize what xformers profiler provide us.

     - Visualize GPU utilization
          - Log memory, GPU utilization...

- [ ] Async creation of the next batch as forward and backward passes are being done.
     - We could do it manually with multiprocessing.
     - We could also use pytorch native dataloader. This is simpler, we should go for this solution.
     - To optimize the data loading process we need to look at computing performance metrics.

- [ ] GSSM:
Add a special argument that can take 4 arguments. `None`: keep the same logic as now. `transition`: transition matrix change for each generation. `slow` the argmax of the transition is the diagonal (argmax p(y | x) = x). `dead` the argmax of the transition is a column (argmax p(y | x) = c).

- [ ] Dataloader: split between train and test. (Probably write to file a train and test set, and read these).


- [ ] Probing.

- [ ] Slurm array option to cross-validate learning rates.

- [ ] Evaluation : can probing recover the hidden state.
- [ ] Generation: Implement caching mechanisms from meta-lingua.

- [ ] Model parallelism

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
python -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml launcher=bash
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml torchrun=True
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml torchrun=True launcher=bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml

## Longer term project

Formal math:
xlean
Metagen -> Lean formal ...


