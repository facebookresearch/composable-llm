# Second experiment

**Question:**
Does a transformer do better with a lot of small independent latent variables or a big one?

**Layman Question:**
Is a transformer particularly good at leveraging latents mechanism?

**How to interpret the results:**
Many nodes mean many independent mechanisms.

## Experiment order
Choose a emission concentration parameter `alpha_X`.

#### Set difficulty level
Then determine some equivalent `alpha_Z` for various graphs.
This could be done by running
```bash
python -m src.apps.gssm.difficulty src/apps/gssm/configs/experiment2/difficulty.yaml
```
Or run on the cluster with
```bash
sbtach src/apps/gssm/configs/experiment2/difficulty.sh
```

#### Run with infinite data
After choosing two pairs, you can generate a training run where you generate new data on the fly.
```bash
python -m src.apps.gssm_onfly.train src/apps/gssm/configs/experiment2/onfly_four_nodes.yaml
```
Or run it the cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/onfly_four_nodes.yaml
```
This configuration runs with four nodes, you can adjust it to test one, two or eight nodes.

#### Generate finite data
You may equally fix the number of data in advance by running
```bash
python -m src.apps.gssm.data src/apps/gssm/configs/experiment2/data.yaml
```
Before running the training with these data
```bash
python -m src.apps.gssm.train src/apps/gssm/configs/experiment2/four_nodes.yaml
```
You can also run it on the cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/four_nodes.yaml
```
