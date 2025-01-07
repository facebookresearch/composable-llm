# TODOS

#### Vivien's Current TODO

- [ ] Make a simple scheduler that keep a constant learning rate.
    - log the learning rate

- [ ] Write visualization scripts/notebooks to visualize logs without wandb.
    - Option to log metrics to wandb after the end of a run.

- [ ] Implement a probe to log activations, and so on.

#### Scaling law with respect to model size

#### Scaling law with respect to the number of data

#### Further improvements
Improvement for `model.py`
1. Benchmark FlexAttention and use it if providing gain.
1. Add a probing mechanism.
1. Implement the caching mechanism.
1. Extrapolate Rope embeddings for sequences much longer than seen at training.
1. Generate completions of prompts of various length at once through funky masking.

Improvement for `train.py`
1. Add gradient clipping.
1. Add mix precision `torch.amp`.

Improvement for `profiler`
1. Add option to log HFU, MFU...

Improvement for `evaluation`
1. Implement parallelization logic beyond DDP.

Improvmenet for `checkpoint`
1. Save consolitated checkpoints.
1. Load checkpoint before DDP / save it after DDP / or use `import torch.distributed.checkpoint as dcp`.