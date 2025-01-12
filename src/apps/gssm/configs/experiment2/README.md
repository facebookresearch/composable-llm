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

Gzip compression should give us a idea of how hard a problem is.
If the compression ratio is above 98%, we may consider it as an almost impossible problem, as their is not much structure to leverage for learning algorithms to show their strenght.
I would assume that a compression ratio aroudnd 75% would be a good starting point for a hard problem that is still `learnable`, in the sense that various learning algorithm will perform differently.
This 75% could change later if we realize that it was not a good target.

The experiment can be launched locally with
```bash
bash src/apps/gssm/configs/experiment2/difficulty.sh
```
Or on the cluster with
```bash
sbatch src/apps/gssm/configs/experiment2/difficulty.sh
```

One can flag the following configurations:
```yaml
one_node:    {"difficulty": 0.75, "alpha_X": 0.001, "alpha_Z": 0.0087}
two_nodes:   {"difficulty": 0.75, "alpha_X": 0.001, "alpha_Z": 0.058}
four_nodes:  {"difficulty": 0.74, "alpha_X": 0.001, "alpha_Z": 0.14}
eight_nodes: {"difficulty": 0.75, "alpha_X": 0.001, "alpha_Z": 0.267}
```

#### Run with infinite data
After choosing pairs, you can generate a training run where you generate new data on the fly.
You may run it on a cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/onfly_one_node.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/onfly_two_nodes.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/onfly_four_nodes.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/onfly_eight_nodes.yaml
```

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
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/one_node.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/two_nodes.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/four_nodes.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment2/eight_nodes.yaml
```
