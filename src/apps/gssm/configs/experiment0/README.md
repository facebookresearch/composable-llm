# Calibration Experiments

**Question:**
What is the right range of hyperparameter for learning? And for GPU utilization?

## Experiment order

#### First experiments: GPU utilization
Maximize the batch size to utilize fully memory, this would ensure high GPU utilization.
You can check GPU utilization through the pytorch profiler (called `heavy profiler` in this codebase).

To choose a model size, recall GPT2 specifications: 
```yaml
gpt2-small:   {n_layer: 12, n_head: 12, n_embd:  768}, # 124M params
gpt2-medium:  {n_layer: 24, n_head: 16, n_embd: 1024}, # 350M params
gpt2-large:   {n_layer: 36, n_head: 20, n_embd: 1280}, # 774M params
gpt2-xl:      {n_layer: 48, n_head: 25, n_embd: 1600}, # 1558M params
vocab size: 50304 
```

Commands:
```bash
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/utilization.yaml
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment0/utilization.yaml
```

#### Second experiments: Learning range

First grid run to look for a good range of hyperparameters.
How big the model should be to learn? How many optimization steps are needed?
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment0/utilization.yaml
```
I found that 4 heads per layer, 4 layers, 64 embedding dimensions and a learning rate of 1e-2 work well.
