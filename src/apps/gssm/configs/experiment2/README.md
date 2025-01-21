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

You can launch an array job to train models
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/nb_data.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/nb_params.yaml
```