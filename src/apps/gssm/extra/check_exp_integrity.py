# %%
import os
from pathlib import Path

# The things that need to have happened
# 1. data available
# 2. entropy calc
# 3. (gzip calc)
# 4. all experiment present

exp_id = int(input("which exp? "))

data_map_path = Path(f"src/apps/gssm/configs/experiment{exp_id}/.gssm_id_path.jsonl")
data_path = Path(f"/checkpoint/nolte/datasets/icml/exp{exp_id}")
logs_path = Path(f"/checkpoint/nolte/icml/logs/exp{exp_id}/")
entropy_path = logs_path / "entropy"

#data exps
nb_data_path = logs_path / "data"
nb_data_logs_path = nb_data_path / "logs"
nb_data_tasks_path = nb_data_path / "tasks"

#param exps
nb_params_path = logs_path / "params"
nb_params_logs_path = nb_params_path / "logs"
nb_params_tasks_path = nb_params_path / "tasks"

#gzip and hmm summary
hmm_summary_path = logs_path / "hmm.jsonl"
gzip_summary_path = logs_path / "gzip.jsonl"


# %%

def check_data(n_data_expected):
  good = True
  if not os.path.exists(data_path):
    print("No data for this experiment")
    return False

  for i in range(n_data_expected):
    dir = data_path / str(i)
    for h5file in ["trainset.h5", "testset.h5"]:
      if not (dir / h5file).exists():
        print(f"missing {h5file} at {dir}")
        good = False

  if good:
    print("all data there")
  return good

def check_entropy(n_data_expected):
  good = True
  if not os.path.exists(entropy_path):
    print("No entropy calc for this experiment")
    return False

  for i in range(n_data_expected):
    dir = entropy_path / str(i)
    if dir.is_dir():
      if not (dir / "eval_0.jsonl").exists():
        print(f"missing entropy calc at {dir}")
        good = False


  if not (gzip_summary_path.exists() and hmm_summary_path.exists()):
    print("missing gzip / hmm summary, run launcher_gzip")
    good = False

  if good:
    print("all entropys / gzip there")
  return good

def check_exps():
  good=True
  paths = [nb_params_logs_path, nb_data_logs_path]
  expected_nums = [n_params_exps_expected, n_data_exps_expected]
  for path,n_expected in zip(paths, expected_nums):
    if not os.path.exists(path):
      print(f"No data for this experiment {path}")
      return False
    count = 0
    for dir in path.iterdir():
      device_output = dir / "device_0.log"
      if not (device_output).exists():
        print(f"missing training at {dir}")
        good=False
        continue

      count += 1
      
      out = device_output.read_text()
      last_line = out.split("\n")[-2]
      if not "Training done" in last_line:
        print(f"training didnt finish at {dir}")
        good=False

    if count != n_expected:
      print(f"missing exps {path},  {count} vs {n_expected}")
      good=False

  if good:
    print("all exps there")
  return good



n_data_expected = len(data_map_path.read_text().split("\n")) - 1 # this many jsonl entries
print(f"expecting {n_data_expected} data files")
check_data(n_data_expected)
check_entropy(n_data_expected)

n_data_exps_expected = len(tuple(nb_data_tasks_path.iterdir()))
n_params_exps_expected = len(tuple(nb_params_tasks_path.iterdir()))
print(f"expecting {n_data_exps_expected} data exp, {n_params_exps_expected} params exp")
check_exps()
# %%
