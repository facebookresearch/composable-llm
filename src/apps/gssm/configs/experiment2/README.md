# Second experiment

**Question:**
Does a transformer do better with a lot of small independent latent variables or a big one?

**Layman Question:**
Is a transformer particularly good at leveraging latents mechanism?

**How to interpret the results:**
Many nodes mean many independent mechanisms.

## Experiment order
Start by defining some graphs, and generating data
```bash
bash src/apps/gssm/configs/experiment2/data.sh
```
You can also launch this job on the cluster
```bash
sbatch src/apps/gssm/configs/experiment2/data.sh
```

#### (TODO) Generate finite data
Before running the training with these data
```bash
python -m src.apps.gssm.train src/apps/gssm/configs/experiment2/four_nodes.yaml
```
You can also run it on the cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/one_node.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/two_nodes.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/four_nodes.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/eight_nodes.yaml
```
