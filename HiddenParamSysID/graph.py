import os
from typing import List

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

def parse_tensorboard(path, scalars:List[str]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

logdir = "./logs/predictors"

run_data = {}
step = None
for run_dir in reversed(sorted(os.listdir(logdir))):
    if os.path.isfile(os.path.join(logdir, run_dir)):
        continue

    # fetch run type by reading from file.
    child_dir = os.listdir(os.path.join(logdir, run_dir))[0]
    alg_type = None
    if child_dir == "Transformer":
        alg_type = "Transformer"
    elif child_dir == "MLP":
        alg_type = "MLP"
    elif child_dir == "FE":
        alg_type = "FE"
    elif child_dir == "MLPOracle":
        alg_type = "MLPOracle"
    elif child_dir == "FE_Dif":
        alg_type = "FE_Dif"
    else:
        print("Skipping run dir", run_dir)
        continue
    if alg_type not in run_data:
        run_data[alg_type] = []

    # read data, convert to numpy array
    try:
        data = parse_tensorboard(os.path.join(logdir, run_dir, child_dir), ["test/loss"])
    except Exception:
        print("Skipping ", logdir, run_dir, child_dir)
        continue
    if step is None or len(data["test/loss"]['step'].to_numpy()) > len(step):
        step = data["test/loss"]['step'].to_numpy()
    data = data["test/loss"]['value'].to_numpy()

    # add to dict
    run_data[alg_type].append(data)


# now plot
# for each alg type in the run data, compute median and 1st and 3rd quartile
fig, ax = plt.subplots()
for alg_type in run_data:
    alg_data = run_data[alg_type]
    length = alg_data[1].shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]

    alg_data = np.stack(alg_data, axis=0)
    print(alg_type, alg_data.shape)
    quartiles = np.quantile(alg_data, [0.0, 0.5, 1.0], axis=0)

    # plot with shaded area
    color = "blue" if alg_type == "FE" else "red" if alg_type == "MLP" else "green" if alg_type == "Transformer" else "purple" if alg_type == "MLPOracle" else "black"
    ax.plot(step, quartiles[1], label=alg_type, color=color)
    ax.fill_between(
        step,
        quartiles[0],
        quartiles[2],
        alpha=0.2,
        color=color,
    )

ax.set_xlabel("Gradient Steps")
ax.set_ylabel("MSE")
ax.legend()
ax.set_ylim([0, 0.4])
plt.savefig(os.path.join(logdir, "graph.png"), dpi=400)


# write quartiles to csv for latex plotting, uses pandas to do one alg at a time
df = pd.DataFrame(step, columns=["step"])
for alg_type in run_data:
    alg_data = run_data[alg_type]
    length = alg_data[1].shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]
    alg_data = np.stack(alg_data, axis=0)
    quartiles = np.quantile(alg_data, [0.0, 0.5, 1.0], axis=0)
    df[alg_type + "_min"] = quartiles[0]
    df[alg_type + "_median"] = quartiles[1]
    df[alg_type + "_max"] = quartiles[2]

df.to_csv(os.path.join(logdir, "quartiles.csv"), index=False)