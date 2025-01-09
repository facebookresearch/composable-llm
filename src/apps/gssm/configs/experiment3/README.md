# Second experiment

**Question:**
Does a transformer do better with a independent latent variables?

**Layman Question**
Does a transformer benefit of mechanisms/subskills to be independent?

**How to interpret the results:**
Edges between nodes mean dependent mechanisms.

## Experiment order
First choose a graph. By default, I would set it to be (with our configuration notation):
```yaml
gssm:
    - name: Z1
      state_dim: 2
      parents: [X]
    - name: Z2
      state_dim: 2
      parents: [X]
    - name: Z3
      state_dim: 2
      parents: [X]
    - name: Z4
      state_dim: 2
      parents: [X]
    - name: Z5
      state_dim: 2
      parents: [X]
    - name: Z6
      state_dim: 2
      parents: [X]
    - name: Z7
      state_dim: 2
      parents: [X]
    - name: Z8
      state_dim: 2
      parents: [X]
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8]
      alpha: 1e-3
```
Then add edges
```yaml
gssm:
    - name: Z1
      state_dim: 2
      parents: [X]
    - name: Z2
      state_dim: 2
      parents: [X, Z1]
    - name: Z3
      state_dim: 2
      parents: [X]
    - name: Z4
      state_dim: 2
      parents: [X, Z3]
    - name: Z5
      state_dim: 2
      parents: [X]
    - name: Z6
      state_dim: 2
      parents: [X, Z5]
    - name: Z7
      state_dim: 2
      parents: [X]
    - name: Z8
      state_dim: 2
      parents: [X, Z7]
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8]
      alpha: 1e-3
```
And more
```yaml
gssm:
    - name: Z1
      state_dim: 2
      parents: [X]
    - name: Z2
      state_dim: 2
      parents: [X, Z1]
    - name: Z3
      state_dim: 2
      parents: [X, Z2]
    - name: Z4
      state_dim: 2
      parents: [X, Z3]
    - name: Z5
      state_dim: 2
      parents: [X, Z4]
    - name: Z6
      state_dim: 2
      parents: [X, Z5]
    - name: Z7
      state_dim: 2
      parents: [X, Z6]
    - name: Z8
      state_dim: 2
      parents: [X, Z7]
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8]
      alpha: 1e-3
```
And more
```yaml
gssm:
    - name: Z1
      state_dim: 2
      parents: [X]
    - name: Z2
      state_dim: 2
      parents: [X, Z1]
    - name: Z3
      state_dim: 2
      parents: [X, Z1, Z2]
    - name: Z4
      state_dim: 2
      parents: [X, Z1, Z2, Z3]
    - name: Z5
      state_dim: 2
      parents: [X, Z1, Z2, Z3, Z4]
    - name: Z6
      state_dim: 2
      parents: [X, Z1, Z2, Z3, Z4, Z5]
    - name: Z7
      state_dim: 2
      parents: [X, Z1, Z2, Z3, Z4, Z5, Z6]
    - name: Z8
      state_dim: 2
      parents: [X, Z1, Z2, Z3, Z4, Z5, Z6, Z7]
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8]
      alpha: 1e-3
```

Find the equivalent `alpha_Z` and perform the same steps as before.

Interestingly, it seems that the compression rate does not depend on the edges.

#### Run with infinite data
After choosing pairs, you can generate a training run where you generate new data on the fly.
You may run it on a cluster with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/onfly_easy.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/onfly_medium.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/onfly_hard.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment3/onfly_dense.yaml
```
