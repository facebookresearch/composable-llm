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
Start by defining some graphs, and generating data
```bash
bash src/apps/gssm/configs/experiment1/data.sh
```
You can also launch this job on the cluster
```bash
sbatch src/apps/gssm/configs/experiment1/data.sh
```

#### (Deprecated) Infinite data
Commands to launch training runs where new data are generated on the fly.
```bash
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/experiment1/onfly.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/onfly.yaml
```

#### (TODO) Finite data
Start by generating data in advance by running
```bash
python -m src.apps.gssm.data src/apps/gssm/configs/experiment1/data.yaml
```
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/from_file.yaml
```

#### (TODO) Compression performance


#### Note on setting the difficulty level
How to determine fair `alpha` for various graphs?

Gzip compression should give us a idea of how hard a problem is.
If the compression ratio is above 98%, we may consider it as an almost impossible problem, as their is not much structure to leverage for learning algorithms to show their strenght.
I would assume that a compression ratio aroudnd 75% would be a good starting point for a hard problem that is still `learnable`, in the sense that various learning algorithm will perform differently.
This 75% could change later if we realize that it was not a good target.