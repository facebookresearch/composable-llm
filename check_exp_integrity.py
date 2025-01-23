# %%
import os
from pathlib import Path

# The things that need to have happened
# 1. data available
# 2. entropy calc
# 3. (gzip calc)
# 4. all experiment present

exp_id = int(input("which exp? "))

data_map_path = Path(f"src/apps/gssm/configs/experiment{exp_id}/map_grid_id_gssm_id.jsonl")
data_path = Path(f"/checkpoint/nolte/datasets/icml/exp{exp_id}")
logs_path = Path(f"/checkpoint/nolte/icml/logs/exp{exp_id}/")
entropy_path = logs_path / "entropy"

nb_data_path = logs_path / "data"
nb_data_logs_path = nb_data_path / "logs"
nb_data_tasks_path = nb_data_path / "tasks"

nb_params_path = logs_path / "params"
nb_params_logs_path = nb_params_path / "logs"
nb_params_tasks_path = nb_params_path / "tasks"

n_data_exps_expected = len(tuple(nb_data_tasks_path.iterdir()))
n_params_exps_expected = len(tuple(nb_params_tasks_path.iterdir()))
n_data_expected = len(data_map_path.read_text().split("\n")) - 1 # this many jsonl entries
print(f"expecting {n_data_exps_expected} data exp, {n_params_exps_expected} params exp, {n_data_expected} data files")
# %%

def check_data():
  good = True
  if not os.path.exists(data_path):
    print("No data for this experiment")
    return False
  count = 0
  for dir in data_path.iterdir():
    count += 1
    for h5file in ["trainset.h5", "testset.h5"]:
      if not (dir / h5file).exists():
        print(f"missing {h5file} at {dir}")
        good = False

  if count != n_data_expected:
    print(f"missing data {count} vs {n_data_expected}")
    good=False

  if good:
    print("all data there")
  return good

def check_entropy():
  good = True
  if not os.path.exists(entropy_path):
    print("No entropy calc for this experiment")
    return False

  count = 0

  for dir in entropy_path.iterdir():
    if dir.is_dir() and dir.name != "tasks":
      count += 1
      if not (dir / "eval_0.jsonl").exists():
        print(f"missing entropy calc at {dir}")
        good = False

  if count != n_data_expected:
    print(f"missing entropy folder {count} vs {n_data_expected}")
    good=False

  if good:
    print("all entropys there")
  return good

def check_exps():
  good=True
  paths = [nb_params_logs_path, nb_data_logs_path]
  expected_nums = [n_params_exps_expected, n_data_exps_expected]
  for path,n_expected in zip(paths, expected_nums):
    if not os.path.exists(path):
      print("No nb_params data for this experiment")
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

check_data()
check_entropy()
check_exps()
# %%
