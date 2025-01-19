# Second experiment

**Question:**
Does a transformer do better with slow evolving features, or with a dead mode?

## Experiment order
Choose a graph. Set some nodes in various modes: slow evolution (`slow`), return to default state (`dead`), with transition matrix sample for each new sequences (`context`).

Experiments can be launched with
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/onfly.yaml
```
(TODO) As well as
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/base.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/slow.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/dead.yaml
python -m src.nanollama.launcher src/apps/gssm/configs/experiment4/context.yaml
```