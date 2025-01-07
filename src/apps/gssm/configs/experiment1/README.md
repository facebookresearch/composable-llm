# Second experiment

**Question:**
Does a transformer do better with a lot of small independent latent variables or a big one?

**How to interpret the results:**
Many nodes mean many independent mechanisms.

## Experiment order
Choose a emission concentration parameter `alpha_X`.

#### Set difficulty level
Then determine some equivalent `alpha_Z` for various graphs.
This could be done by running
```bash
python -m src.apps.gssm.difficulty src/apps/experiment2/difficulty.yaml
```
Or run on the cluster with
```bash
sbtach src/apps/experiment2/difficulty.sh
```

#### Run with infinite data
After choosing two pairs, you can generate a training run where you generate new data on the fly.
```bash
python -m src.apps.gssm_onfly.train src/apps/experiment1/onfly_small_X.yaml
python -m src.apps.gssm_onfly.train src/apps/experiment1/onfly_small_Y.yaml
```
Or run it the cluster with
```bash
python -m src.nanollama.launcher src/apps/experiment1/onfly_small_X.yaml
python -m src.nanollama.launcher src/apps/experiment1/onfly_small_Z.yaml
```

#### Generate finite data
You may equally fix the number of data in advance by running
```bash
python -m src.apps.gssm.data src/apps/experiment1/data.yaml
```
Before running the training with these data
```bash
python -m src.apps.gssm.train src/apps/experiment1/small_X.yaml
python -m src.apps.gssm.train src/apps/experiment1/small_Z.yaml
```
You can also run it on the cluster with
```bash
python -m src.nanollama.launcher src/apps/experiment1/small_X.yaml
python -m src.nanollama.launcher src/apps/experiment1/small_Z.yaml
```