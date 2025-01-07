# Second experiment

**Question:**
Does a transformer do well for hard in context learning task?

#### Choose a graph
First choose a graph. By default, I would set it to be (with our configuration notation):
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
      alpha: 1e-3
```
Set the nodes to be in context-learning mode
```yaml
gssm:
    - name: Z1
      state_dim: 4
      parents: [X]
      mode: context
    - name: Z2
      state_dim: 4
      parents: [X]
      mode: context
    - name: Z3
      state_dim: 4
      parents: [X]
      mode: context
    - name: Z4
      state_dim: 4
      parents: [X]
      mode: context
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4]
      alpha: 1e-3
```
Find the equivalent `alpha_Z` and perform the same steps as before.
