# Second experiment

**Question:**
Does a transformer do better with a independent latent variables?

**Layman Question**
Does a transformer benefit of mechanisms/subskills to be independent?

**How to interpret the results:**
Edges between nodes mean dependent mechanisms.

## Experiment order
Start by defining some graphs, and generating data
```bash
bash src/apps/gssm/configs/experiment3/data.sh
```
You can also launch this job on the cluster
```bash
sbatch src/apps/gssm/configs/experiment3/data.sh
```