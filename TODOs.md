
- Fix issue regarding TrainState.state_dict not working, while the same logic seems to work MetaLingua.

Things to be mindful for Parallelization:
- TrainState(Stateful)
- CheckpointManager.dp_rank