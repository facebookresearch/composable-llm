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
bash src/apps/gssm/configs/experiment9/data.sh
```
You can also launch this job on the cluster
```bash
sbatch src/apps/gssm/configs/experiment9/data.sh
```

You can launch an array job to train models
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment9/nb_data.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment9/nb_params.yaml
```

#### Entropy baselines
You can launch entropy estimate with 
```bash
python -m src.apps.gssm.launcher_entropy
```
It will write some bash script that are going to be launched with `sbatch`.

Once the entropy estimates are done, you should merge the results into a single file which is actually done in the following file
```bash
python -m src.apps.gssm.launcher_gzip
```
This file also compute an estimate of the entropy from the compression one can get with gzip.
Feel free to command some part of the file to launch only the part you want.
