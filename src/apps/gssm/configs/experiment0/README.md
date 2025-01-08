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

First I debug it locally on P100 GPUs.
```
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/utilization_small.yaml
```

I will launch this on V100 GPUs.


Learn first with infinite data.
Check how many optimization steps, which learning rates, are needed to converge, and what model size is needed to achieve the lowest possible error.

Repeat the experiments with fixed dataset size.

#### Play with various graphs and models size
This is to check that the hyperparameters work well in various settings.

<!-- #### Set difficulty level
Then determine some equivalent pairs for `(alpha_X, alpha_Z)` with a small `alpha_X` and a big `alpha_Z`, and a big `alpha_X` and a small `alpha_Z`.
This could be done by running
```bash
python -m src.apps.gssm.difficulty src/apps/gssm/configs/experiment1/difficulty.yaml
```
Or run on the cluster with
```bash
sbatch src/apps/gssm/configs/experiment1/difficulty.sh
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

#### Generate finite data
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
``` -->
