# TODOS

#### Scaling law with respect to model size

#### Scaling law with respect to the number of data

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
1. Add option to log HFU, MFU...
1. Write a script to cast the profiler traces to numpy array to visualize them, and to wandb.

Improvement to `wanbd`
1. Do not log to wanbd, first log to local file, and have a asynchronous process casting the logs to wandb (or to another visualizer).
