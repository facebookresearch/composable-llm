# Second experiment

**Question:**
Does a transformer do better with slow evolving features, or with a dead mode?

## Experiment order
Start by defining some graphs, and generating data
```bash
bash src/apps/gssm/configs/experiment4/data.sh
```
You can also launch this job on the cluster
```bash
sbatch src/apps/gssm/configs/experiment4/data.sh
```

You can launch an array job to train models
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/nb_data.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/nb_params.yaml
```