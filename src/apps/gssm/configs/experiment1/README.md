# First experiment

**Question:**
Does a transformer do better with a small alpha_X and a big alpha_Z, or a big alpha_X and a small alpha_Z?

**Layman Question:**
Does a transformer prefer easy to infer latents, or easy to predict latents?

**How to interpret the results:**
A small alpha_X mean "easy to infer the latent variables from observations"
A small alpha_Z mean "easy to predict the evolution of the latent variables" 

**Caveat:**
Does the result of the experiments depend on the graph?

## Experiment order
#### Choose a graph
First choose a graph. By default, I have set it to be (with our configuration notation):
```yaml
gssm:
    - name: Z1
      state_dim: 4
      parents: [X]
    - name: Z2
      state_dim: 4
      parents: [X]
    - name: Z3
      state_dim: 4
      parents: [X]
    - name: Z4
      state_dim: 4
      parents: [X]
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4]
```

#### Set difficulty level
Then determine some equivalent pairs for `(alpha_X, alpha_Z)` with a small `alpha_X` and a big `alpha_Z`, and a big `alpha_X` and a small `alpha_Z`.

Gzip compression should give us a idea of how hard a problem is.
If the compression ratio is above 98%, we may consider it as an almost impossible problem, as their is not much structure to leverage for learning algorithms to show their strenght.
I would assume that a compression ratio aroudnd 75% would be a good starting point for a hard problem that is still `learnable`, in the sense that various learning algorithm will perform differently.
This 75% could change later if we realize that it was not a good target.

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

#### Maximize GPU utilization
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
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment1/onfly_small_Z.yaml
```
With a eight GPUs
```bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 8 -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment1/onfly_small_Z.yaml
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

#### Run with infinite data
After choosing two pairs, you can generate a training run where you generate new data on the fly.
```bash
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment1/onfly_small_X.yaml
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment1/onfly_small_Z.yaml
```
Or run it the cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/onfly_small_X.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/onfly_small_Z.yaml
```

#### Run with finite data (TODO)
You may equally fix the number of data in advance by running
```bash
python -m src.apps.gssm.data src/apps/gssm/configs/experiment1/data.yaml
```
Before running the training with these data
```bash
python -m src.apps.gssm.train src/apps/gssm/configs/experiment1/small_X.yaml
python -m src.apps.gssm.train src/apps/gssm/configs/experiment1/small_Z.yaml
```
You can also run it on the cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/small_X.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/small_Z.yaml
```
