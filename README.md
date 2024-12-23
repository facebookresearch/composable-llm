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

## First run
You can run a debug script with the following command:
```
python -m src.apps.gssm.train config=src/apps/gssm/configs/debug.yaml
```
To run some real experiments:
```
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/base.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/sparse.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/dense.yaml
python -m src.nanollama.launcher script=src.apps.gssm.train config=src/apps/gssm/configs/low_entropy.yaml
```

## Development
For formatting, I recommand using `black`, `flake8`, and `isort`.
Consider automatic formatting when saving files (easy to setup in VSCode, ask ChatGPT to get set up if not confortable with VSCode configuration).

## Organization
The main code is in the `src` folder.
The base functions are in `src.nanollama` (supposed not to change much over time) and the application specific code is in `src.apps` (suppose to change for your specific applications).
