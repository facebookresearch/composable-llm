# Calibration Experiments

**Question:**
What is the right range of hyperparameter for learning? And for GPU utilization?

## Experiment order

#### First experiments: GPU utilization
Maximize the batch size to utilize fully memory, this would ensure high GPU utilization.
You can check GPU utilization through the pytorch profiler (called `heavy profiler` in this codebase).

I will take the biggest model I may consider with 12 layers, 12 heads per layer, 256 embedding dimension, and four times this for the hidden dimension.
Recall GPT2 specifications: 
```yaml
gpt2-small:   {n_layer: 12, n_head: 12, n_embd:  768}, # 124M params
gpt2-medium:  {n_layer: 24, n_head: 16, n_embd: 1024}, # 350M params
gpt2-large:   {n_layer: 36, n_head: 20, n_embd: 1280}, # 774M params
gpt2-xl:      {n_layer: 48, n_head: 25, n_embd: 1600}, # 1558M params
vocab size: 50304 
```

First I debug it locally on P100 GPUs, with a single GPU
```bash
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/onfly_small.yaml
```
With a two GPUs
```bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/onfly_small.yaml
```
I found the that the following saturate memory, and lead to 98% GPU utilization (which I check by visualizing the traces of the heavy profiler with tensorboard):
```yaml
cluster:
    mem: 16G
data:
    seq_len: 2048
    batch_size: 16
model:
    vocab_size: 32
    emb_dim: 288
    nb_layers: 12
    nb_heads: 12
```

#### Second experiments: Learning range

First grid run to look for a good range of hyperparameters.
How big the model should be to learn? How many optimization steps are needed?
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment0/onfly_small.yaml
```
I found that 4 heads per layer, 4 layers, 64 embedding dimensions and a learning rate of 1e-2 work well.
