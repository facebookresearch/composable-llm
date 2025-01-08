# TODOS

Compute entropy in close form.
Think about the entropy baseline for "in-context learning" experiments.

#### Further improvements

Generation:
1. Implement the caching mechanism.
1. Generate completions of prompts of various length at once through funky masking.

Long context:
1. Extrapolate Rope embeddings for sequences much longer than seen at training.

Optimization
1. Add gradient clipping.
1. Test gradient accumulation.

Faster training:
1. Add mix precision `torch.amp`.
1. Benchmark FlexAttention and use it if providing gain.

Profiling:
1. Add option to log HFU, MFU...
1. Add a probing mechanism.

Parallelization
1. Implement parallelization logic beyond DDP.
1. Save consolitated checkpoints.

Visualization:
1. Add local visualization tools.
1. Add option to log to wandb after the end of a run.
