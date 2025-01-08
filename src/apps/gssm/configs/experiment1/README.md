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
#### Choose a graph
First choose a graph. By default, I have set it to be (with our configuration notation):
```yaml
gssm:
    - name: Z1
      state_dim: 4
      parents: [X]
    - name: Z2
      state_dim: 4
      parents: [X]
    - name: Z3
      state_dim: 4
      parents: [X]
    - name: Z4
      state_dim: 4
      parents: [X]
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4]
```

#### Set difficulty level
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
python -m src.apps.gssm.train_on_fly src/apps/gssm/configs/experiment1/onfly_small_X.yaml
python -m src.apps.gssm.train_on_fly src/apps/gssm/configs/experiment1/onfly_small_Z.yaml
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
```
