# Second experiment

**Question:**
Does a transformer do better with slow evolving features, or with a dead mode?

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
Set the nodes to be in slow evolving mode
```yaml
gssm:
    - name: Z1
      state_dim: 4
      parents: [X]
      mode: slow
    - name: Z2
      state_dim: 4
      parents: [X]
      mode: slow
    - name: Z3
      state_dim: 4
      parents: [X]
      mode: slow
    - name: Z4
      state_dim: 4
      parents: [X]
      mode: slow
    - name: X
      state_dim: 32
      parents: [Z1, Z2, Z3, Z4]
      alpha: 1e-3
```
Set the nodes to be in dead mode
```yaml
gssm:
    - name: Z1
      state_dim: 4
      parents: [X]
      mode: dead
    - name: Z2
      state_dim: 4
      parents: [X]
      mode: dead
    - name: Z3
      state_dim: 4
      parents: [X]
      mode: dead
    - name: Z4
      state_dim: 4
      parents: [X]
      mode: dead
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
