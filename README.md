# From Nodes to Narratives: What Type of Learners Transformers Are?

Unveiling structural assumptions that transformers leverage.

## Installation

The code runs Python 3.10+.
Here is some installation instruction:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Follow the instruction online, most likely you will execute the following commands.
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
- Install python in a new conda environment: be mindful to install a version of python that is compatible with PyTorch.
```bash
conda create -n llm
conda activate llm
conda install pip python=3.12
```
- Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)) (use `nvidia-smi` to check your CUDA driver)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```
This should print "True".
- Install this repo
```bash
git clone <repo url>
cd <repo path>
pip install -e .
```
If you want to install the development, visualization and mamba dependencies, you can swap the previous command for the following one:
```bash
pip install -e .[dev,mamba,visu]
```

For mamba, `conv_1d` can be a bit hard to load. You may want to load environment variables with:
```bash
module load cuda/12.1
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
pip install -e .[mamba]
```

## Using the codebase

#### First run
You can run a debug script with the following command:
```bash
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/debug/onfly.yaml
```
In this online setting, data are generated on the fly. 
The configuration file `debug.yaml` is a simple configuration file that will run a simple experiment.

#### Data
The previous config will generate batch of data on the fly. Each new batch will be made of new data.
If you want to generate data offline first, and only read from these datasets, you can do so with the following command to generate datasets:
```bash
python -m src.apps.gssm.data src/apps/gssm/configs/debug/data.yaml
```
Here the `data.yaml` is a configuration files to generate various datasets.
You can modify this configuration to suit your needs.

#### Cluster runs
You may launch experiments to run on a cluster with the following command:
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/debug/train.yaml
```
This will launch them on a slurm cluster (assuming your are logged into one).
You equally run experiments on multi-gpu locally with torchrun, e.g.,
```bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 -m src.apps.gssm.train src/apps/gssm/configs/debug/train.yaml
```
You can equally run an array job is you specify a grid to iterate over in your config file.

Running job on a cluster will create a `logging` folder defined by the `log_dir` arguments in the configuration file (by default, it goes to `$HOME/logs`).
The logging directory is organized in the following fashion.
```
root/
├── checkpoint/: models (and dataloaders, ...) checkpoints
├── code/: copy of the codebase when launching the job
├── evals/: evaluation tasks
├── logs/: run logs
├── metrics/: metrics logged during the runs
├── run.sh: the script that was run
├── stdout/: logs of the stdout
└── tasks/{id}.yaml: configuration files
```

#### Command summary
To summarize the commands, for training while generating data on the fly:
```bash
python -m src.apps.gssm.train_onfly src/apps/gssm/configs/debug/onfly.yaml
```
For generating data first, and then training from them:
```bash
python -m src.apps.gssm.train src/apps/gssm/configs/debug/train.yaml
python -m src.apps.gssm.train src/apps/gssm/configs/debug/train.yaml
```
For launching a run on a Slurm cluster:
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/debug/train.yaml
```
For launching an array job on a Slurm cluster:
```bash
python -m src.nanollama.launcher src/apps/gssm/configs/debug/grid.yaml
```

## Development
For formatting, I recommand using `ruff`.
Consider automatic formatting when saving files (easy to setup in VSCode).

#### Organization
The main code is in the `src` folder.
The base functions are in `src.nanollama` (supposed not to change much over time) and the application specific code is in `src.apps` (supposed to change for your specific applications).
The codebase is object-oriented. Your training script will be monitor by a `monitor` object, which is notably composed by a `profiler`, a `logger` and a `checkpointer`.
Other objects include your training `model`, your `optimizer` (and scheduler), and your `data` loader.
Files are organized to follow this logic.
```
root/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   ├── apps/
│   │   └── my_app/
│   │       ├── train: a script to train a model
│   │       └── data: a script to generate a dataset
│   └── nanollama:
│       ├── cluster:
│       │   ├── cluster: utils to get a cluster running your applications
│       │   └── slurm: utils to configure a Slurm job
│       ├── data:
│       │   └── my_data: a class to generate data
│       ├── model:
│       │   └── my_model: a class to define a model
│       ├── monitor:
│       │   ├── checkpoint: a checkpoint manager
│       │   ├── logger: a logger manager
│       │   ├── monitor: main monitoring object
│       │   └── profiler: a profiler manager
│       ├── optim: optimizer and scheduler
│       └── launcher: script to launch a training via Slurm or torchrun
└── tests/
```

## Known bugs
- When running a torchrun with asynchronous data loader, keyboard interrupt are not always intercepted by all gpus.
The GPUs that do not intercept the keyboard interrupt do not exit the context stack, not checkpointing their state.

- Some GPUs do not seem to like the instanciation of SIGUSR1.