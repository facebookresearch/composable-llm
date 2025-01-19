# Second experiment

**Question:**
Does a transformer do better with a independent latent variables?

**Layman Question**
Does a transformer benefit of mechanisms/subskills to be independent?

**How to interpret the results:**
Edges between nodes mean dependent mechanisms.

## Experiment order

Start by choosing a graph, then add edges.
Find the equivalent `alpha_Z` and perform the same steps as before.
Interestingly, it seems that the compression rate does not depend on the edges.

#### Run
After choosing pairs, you can generate a training run where you generate new data on the fly.
You may run it on a cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/onfly.yaml
```

(TODO) Or with finite data
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/easy.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/medium.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/hard.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/dense.yaml
```
