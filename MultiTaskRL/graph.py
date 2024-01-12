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

logdir = "./data/"

run_data_distance = {}
run_data_success_rate = {}
step = None
for run_dir in reversed(sorted(os.listdir(logdir))):
    if run_dir == "graph.png" or run_dir == "data.csv" or run_dir == "multi_task_data.png" or run_dir == "multi_task_data.csv":
        continue
    # fetch run type by reading from file.
    alg_type = None
    if os.path.isfile(os.path.join(logdir, run_dir, "transformer.txt")):
        alg_type = "Transformer"
        sr_tag = "rl/success_rate"
    elif os.path.isfile(os.path.join(logdir, run_dir, "transformer2.txt")):
        alg_type = "Transformer2"
        sr_tag = "rl/success_rate"
    elif os.path.isfile(os.path.join(logdir, run_dir, "reward_encoder.txt")):
        alg_type = "FE"
        sr_tag = "rl/success_rate"
    elif os.path.isfile(os.path.join(logdir, run_dir, "reward_encoder_ablation.txt")):
        alg_type = "FEA"
        sr_tag = "rl/success_rate"
    elif os.path.isfile(os.path.join(logdir, run_dir, "her_dqn.txt")):
        alg_type = "HerDQN"
        sr_tag = "rl/success_rate"
    else:
        alg_type = "FB"
        sr_tag = "rl/total_reward"
    if alg_type not in run_data_distance:
        run_data_distance[alg_type] = []
        run_data_success_rate[alg_type] = []

    # read data, convert to numpy array
    # print("reading", run_dir, " for ", sr_tag)
    try:
        data = parse_tensorboard(os.path.join(logdir, run_dir), ["rl/final_distance", sr_tag])
    except AssertionError as e:
        print(e)
        print(run_dir)
        exit(1)
    if step is None or len(data["rl/final_distance"]['step'].to_numpy()) > len(step):
        step = data["rl/final_distance"]['step'].to_numpy()
    distance = data["rl/final_distance"]['value'].to_numpy()
    success_rate = data[sr_tag]['value'].to_numpy()

    # add to dict
    run_data_distance[alg_type].append(distance)
    run_data_success_rate[alg_type].append(success_rate)


# Multiply step by 2.5 to get it in env interactions instead of updates
step = step * 2.5

# now plot
# for each alg type in the run data, compute median and 1st and 3rd quartile
fig, axs = plt.subplots(1,2, figsize=(10,4))

# plot final distance on left
ax = axs[0]
for alg_type in run_data_distance:
    alg_data = run_data_distance[alg_type]
    length = step.shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]

    if len(alg_data) < 3:
        continue
    alg_data = np.stack(alg_data, axis=0)
    quartiles = np.quantile(alg_data, [0.25, 0.5, 0.75], axis=0)

    # plot with shaded area
    color = "blue" if alg_type == "FE" else \
            "red" if alg_type == "FB" else \
            "green" if alg_type == "Transformer" else \
            "cyan" if alg_type == "FEA" else \
            "purple" if alg_type == "HerDQN" else \
            "black"
    ax.plot(step, quartiles[1], label=alg_type, color=color)
    ax.fill_between(
        step,
        quartiles[0],
        quartiles[2],
        alpha=0.2,
        color=color,
    )
ax.set_xlabel("Env Steps")
ax.set_ylabel("Distance to Objective")

# plot success rate on right
ax = axs[1]
for alg_type in run_data_success_rate:
    alg_data = run_data_success_rate[alg_type]
    length = step.shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]
    if len(alg_data) < 3:
        continue
    alg_data = np.stack(alg_data, axis=0)
    print(alg_type, alg_data.shape)
    quartiles = np.quantile(alg_data, [0.25, 0.5, 0.75], axis=0)

    # plot with shaded area
    color = "blue" if alg_type == "FE" else \
            "red" if alg_type == "FB" else \
            "green" if alg_type == "Transformer" else \
            "cyan" if alg_type == "FEA" else \
            "purple" if alg_type == "HerDQN" else \
            "black"
    ax.plot(step, quartiles[1], label=alg_type, color=color)
    ax.fill_between(
        step,
        quartiles[0],
        quartiles[2],
        alpha=0.2,
        color=color,
    )
ax.set_xlabel("Env Steps")
ax.set_ylabel("Success Rate")
ax.legend(loc="upper left", framealpha=0.0)


plt.savefig(os.path.join(logdir, "graph.png"), dpi=400)


# now write it to csv so we can plot it in latex later
df = pd.DataFrame()
df["step"] = step
for alg_type in run_data_distance:
    alg_data = run_data_success_rate[alg_type]
    length = step.shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]
    if len(alg_data) < 3:
        continue
    quartiles = np.quantile(alg_data, [0.25, 0.5, 0.75], axis=0)
    lower, median, upper = quartiles[0], quartiles[1], quartiles[2]
    df[f"success_rate_{alg_type}_lower"] = lower
    df[f"success_rate_{alg_type}_median"] = median
    df[f"success_rate_{alg_type}_upper"] = upper

    alg_data = run_data_distance[alg_type]
    length = step.shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]
    if len(alg_data) < 3:
        continue

    quartiles = np.quantile(alg_data, [0.25, 0.5, 0.75], axis=0)
    lower, median, upper = quartiles[0], quartiles[1], quartiles[2]
    df[f"distance_{alg_type}_lower"] = lower
    df[f"distance_{alg_type}_median"] = median
    df[f"distance_{alg_type}_upper"] = upper

df.to_csv(os.path.join(logdir, "data.csv"), index=False)
