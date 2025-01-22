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
bash src/apps/gssm/configs/experiment5/data.sh
```
You can also launch this job on the cluster
```bash
sbatch src/apps/gssm/configs/experiment1/data.sh
```

You can launch an array job to train models
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/nb_data.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment1/nb_params.yaml
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

#### (TODO) Plots
Preliminary plots can be generated with the `exp_plots` script.

#### Note on setting the difficulty level
How to determine fair `alpha` for various graphs?

Gzip compression should give us a idea of how hard a problem is.
If the compression ratio is above 98%, we may consider it as an almost impossible problem, as their is not much structure to leverage for learning algorithms to show their strenght.
I would assume that a compression ratio aroudnd 75% would be a good starting point for a hard problem that is still `learnable`, in the sense that various learning algorithm will perform differently.
This 75% could change later if we realize that it was not a good target.
