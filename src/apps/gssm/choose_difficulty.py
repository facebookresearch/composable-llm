# %%
import matplotlib.pyplot as plt
import yaml
import os

from apps.gssm.difficulty import DifficultyEstimationConfig, read_results
from nanollama.utils import initialize_nested_object


def main() -> None:
    """
    Launch a difficulty estimation job from configuration file specified by cli argument.

    Usage:
    ```
    python -m --task-id 1 --nb-tasks 4 difficulty src/apps/my_app/my_config.yaml
    ```
    """
    import argparse

    # parse file configuration path and task id
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--task-id", type=int, default=1, help="Task id in the job array.")
    parser.add_argument("--nb-tasks", type=int, default=1, help="Number of tasks in the job array.")
    args = parser.parse_args()
    path = args.config

    # load configuration from file
    with open(os.path.expandvars(path)) as f:
        all_configs = yaml.safe_load(f)

    plt.figure(figsize=(15, 15))

    for name, config in all_configs.items():
        # initialize configuration
        config = initialize_nested_object(DifficultyEstimationConfig, config)
        print(config.path)
        hmm_estimate, gzip_estimate = read_results(config.path)
        plt.scatter([x for x in hmm_estimate.values()], gzip_estimate.values(), alpha=0.3, label=name)
        for (alphas, h_estimate), estimate in zip(hmm_estimate.items(), gzip_estimate.values()):
            # plt.text(h_estimate, gzip_estimate, f"{alphas[0]:.2f},{alphas[1]:.2f}", ha="center", va="center", font
            plt.annotate(
                f"{alphas[1]:.2e}",
                xy=(h_estimate, estimate),
                xytext=(-10, 10),
                textcoords="offset points",
                ha="right",
                va="bottom",
                arrowprops=dict(arrowstyle="->"),
                fontsize=5,
            )
    plt.xlabel("H estimate")
    plt.ylabel("compression rate")
    plt.legend()
    plt.savefig("compr_vs_h.png")


if __name__ == "__main__":
    main()

# %%
