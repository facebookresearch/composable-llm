# TODOS

Compute entropy in close form.
Think about the entropy baseline for "in-context learning" experiments.

- [ ] Make first experimental runs.
- [ ] Find range of hyperparameters for our study.


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