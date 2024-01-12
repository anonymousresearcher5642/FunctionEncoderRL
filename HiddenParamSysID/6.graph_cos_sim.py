import argparse
import os
import json
import random

import numpy
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from MultiSystemIdentification.FE import FE
from MultiSystemIdentification.VariableCheetahEnv import *
import cv2
from tqdm import tqdm, trange
# now create a graph of the cos-sim using matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
# parser = argparse.ArgumentParser(
#                     prog='6.compute_encodings.py',
#                     description='Computes the encodings of the dynamics variables')
# parser.add_argument('--dimensions_to_investigate', type=str, default="torso_length",
#                     help='The dimensions to investigate')
# args = parser.parse_args()
#
# dimensions_to_investigate = args.dimensions_to_investigate


default_values = {  'friction':DEFAULT_FRICTION,
                    'torso_length':DEFAULT_TORSO_LENGTH,
                    'bthigh_length':DEFAULT_BTHIGH_LENGTH,
                    'bshin_length':DEFAULT_BSHIN_LENGTH,
                    'bfoot_length':DEFAULT_BFOOT_LENGTH,
                    'fthigh_length':DEFAULT_FTHIGH_LENGTH,
                    'fshin_length':DEFAULT_FSHIN_LENGTH,
                    'ffoot_length':DEFAULT_FFOOT_LENGTH,
                    'bthigh_gear':DEFAULT_BTHIGH_GEAR,
                    'bshin_gear':DEFAULT_BSHIN_GEAR,
                    'bfoot_gear':DEFAULT_BFOOT_GEAR,
                    'fthigh_gear':DEFAULT_FTHIGH_GEAR,
                    'fshin_gear':DEFAULT_FSHIN_GEAR,
                    'ffoot_gear':DEFAULT_FFOOT_GEAR,
                    }
titles = {  'friction':"Friction",
            'torso_length':"Torso Length",
            'bthigh_length':"Back Thigh Length",
            'bshin_length':"Back Shin Length",
            'bfoot_length':"Back Foot Length",
            'fthigh_length':"Front Thigh Length",
            'fshin_length':"Front Shin Length",
            'ffoot_length':"Front Foot Length",
            'bthigh_gear':"Back Thigh Gear",
            'bshin_gear':"Back Shin Gear",
            'bfoot_gear':"Back Foot Gear",
            'fthigh_gear':"Front Thigh Gear",
            'fshin_gear':"Front Shin Gear",
            'ffoot_gear':"Front Foot Gear",
            }

# directory of policies
data_dir = "data/"
label_font_size = 10
title_font_size = 12

# load the dynamics variables and encodings from data
savedir = os.path.join(data_dir)
# Create a 2x2 grid of subplots
fig = plt.figure(figsize=(5.2*3, 3*3))
gs = GridSpec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.2])
for i, dimensions_to_investigate in enumerate( default_values.keys()):
    x,y  = i//5, i%5
    default_value = default_values[dimensions_to_investigate]
    title = titles[dimensions_to_investigate]
    env_variables = np.load(os.path.join(savedir, dimensions_to_investigate, "dynamics_variables.npy"), allow_pickle=True)
    encodings = torch.tensor(np.load(os.path.join(savedir,dimensions_to_investigate, "encodings.npy")), dtype=torch.float32).to("cuda:0")


    # compute pairwise cos-sim between encodings
    cos_sim = np.zeros((len(env_variables), len(env_variables)))
    for i in range(len(env_variables)):
        for j in range(len(env_variables)):
            cos_sim[i, j] = torch.mean(torch.nn.CosineSimilarity(dim=1)(encodings[i], encodings[j])).item()
    dynamic_variable_values = [env_variables[i][dimensions_to_investigate][0] for i in range(len(env_variables))]


    # create a color map
    mini = 0.88 # np.min(cos_sim).item()
    jet = cm = plt.get_cmap('plasma_r')
    cNorm = colors.Normalize(vmin=mini, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    labels = [f"{(d/default_value):0.2f}" for d in dynamic_variable_values]
    labels = [labels[0], labels[len(labels)//2], labels[-1]]
    cos_sim = cos_sim[::-1, ::]

    # plot the graph
    ax = fig.add_subplot(gs[x,y])
    im = ax.imshow(cos_sim, cmap=jet, norm=cNorm)
    ax.set_yticks([len(cos_sim)-1, len(cos_sim)//2, 0])
    ax.set_xticks([0, len(cos_sim)//2, len(cos_sim)-1])
    ax.set_xticklabels(labels, fontsize=label_font_size)
    ax.set_yticklabels(labels, fontsize=label_font_size)
    # ax.set_xlabel(dimensions_to_investigate.replace("_", " ").title())
    # ax.set_ylabel()
    ax.set_title(title, fontsize=title_font_size)

# make last two blank
ax = fig.add_subplot(gs[2,4])
ax.axis('off')
# ax = fig.add_subplot(gs[3,3])
# ax.axis('off')

cax = fig.add_subplot(gs[:, 5])  # Span all rows in the last column
fig.colorbar(scalarMap, cax=cax)
fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.15, hspace=0.25)
# fig.colorbar(im)
# plt.tight_layout()
plt.savefig(os.path.join(savedir, "cos_sim.png"), dpi=600)