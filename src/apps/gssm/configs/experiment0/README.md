# Calibration Experiments

**Question:**
What is the right range of hyperparameter for learning? And for GPU utilization?

## Experiment order

#### First experiment: Find a hard problem
Gzip compression should give us a idea of how hard a problem is.
If the compression ratio is above 98%, we may consider it as an almost impossible problem, as their is not much structure to leverage for learning algorithms to show their strenght.
I would assume that a compression ratio aroudnd 75% would be a good starting point for a hard problem that is still `learnable`, in the sense that various learning algorithm will perform differently.
This 75% could change later if we realize that it was not a good target.

I first choose a graph.
Take a hard problem (e.g, alpha_Z = 1e-2, alpha_X = 1e-2). 
Check with gzip that we can compress it (compression ration below 75%).

The experiment can be launched locally with
```bash
bash src/apps/gssm/configs/experiment0/difficulty.sh
```
Or on the cluster with
```bash
sbatch src/apps/gssm/configs/experiment0/difficulty.sh
```

For example, with 32 observable tokens and four hidden nodes with four states, one can flag
```yaml
setup1: {alpha_X: 1e-3, alpha_Z: 0.125, difficulty: 0.74},
setup2: {alpha_X: 5e-2, alpha_Z: 1e-3, difficulty: 0.72}
```
While with 64 observable tokens and eight hidden nodes with four states, one can flag
```yaml
setup1: {alpha_X: 1e-3, alpha_Z: 4e-3, difficulty: 0.74},
setup2: {alpha_X: 2e-3, alpha_Z: 1e-3, difficulty: 0.73}
```

#### Second experiments: GPU utilization
Maximize the batch size to utilize fully memory.
This would ensure high GPU utilization.
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
```
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/utilization_small.yaml
```
With a two GPUs
```
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/utilization_small.yaml
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

#### Third experiments: Learning range

First grid run to look for a good range of hyperparameters.
How big the model should be to learn? How many optimization steps are needed?
```
python -m src.nanollama.launcher src/apps/gssm/configs/experiment0/onfly_small.yaml
```
I have started in a setting with 262144 tokens (nb_gpus * batch_size * seq_len * grad_acc = 8 * 16 * 2048 * 1 = 262144) per optimization steps.
I have started with new data being generated for each batch (on the fly).

I will later repeat the experiments with a fixed dataset size, and with a different data generating model.

#### Play with various graphs and models size
This is to check that the hyperparameters work well in various settings.
