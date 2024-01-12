import argparse
import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from MultiSystemIdentification.FE import FE
from MultiSystemIdentification.Get_Data import *
from MultiSystemIdentification.MLP import MLP
from MultiSystemIdentification.MLP_oracle import  MLPOracle
from MultiSystemIdentification.Transformer import Transformer
from MultiSystemIdentification.Ablation.FE_PWN import FE_PWN
from MultiSystemIdentification.Ablation.FE_orthonormalization import FE_orthonormalization
from MultiSystemIdentification.Ablation.FE_F1 import FE_F1
from MultiSystemIdentification.Ablation.FE_Dif import FE_Dif

# add arg parser to read for model type and seed
parser = argparse.ArgumentParser(
                    prog='4.train_predictors.py',
                    description='Trains a predictor of the given type on the transition data')
parser.add_argument('--model_type', type=str, default="FE",
                    help='The type of predictor to train. Options are FE, MLP, and TRANSFORMER')
parser.add_argument('--seed', type=int, default=0,
                    help='The seed to use for the random number generator')
parser.add_argument('--low_data', action='store_true',
                    help='Whether to use low data or not')
args = parser.parse_args()

assert args.model_type in ["FE", "FE_PWN", "FE_ON", "FE_F1", "FE_Dif", "MLP", "TRANSFORMER", "MLPOracle"]
model_type = args.model_type
seed = args.seed

# seed everything
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# input parameters
epochs = 1000
if args.low_data:
    example_data_size = 200 # this is the size the transformer uses because of memory constraints
    # we can beat its performance with the same data using FE_Dif
else:
    example_data_size = 5000


if example_data_size != 5000:
    print("\n\n\n\nWARNING: example_data_size is not 5000")
    print("which is used for the paper. Lower sizes are used for ablations.\n\n\n\n")

batch_size = 50_000
device = "cuda:0"

# make log dir
date_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logdir = f"logs/predictors/{date_time_str}/"
os.makedirs(logdir, exist_ok=True)

# data
(train_xs, train_ys), (testing_xs, testing_ys), input_size, output_size = get_system_data(device)
# (train_xs, train_ys), (testing_xs, testing_ys), input_size, output_size = get_polynomial_data(device) # useful for debugging

# oracle data
hidden_parameters = np.load("data/dyns.npy")

# create predictor
if model_type == "FE":
    predictor = FE(input_size, output_size, embed_size=100, device=device)
elif model_type == "FE_PWN":
    predictor = FE_PWN(input_size, output_size, embed_size=100, device=device)
elif model_type == "FE_ON":
    predictor = FE_orthonormalization(input_size, output_size, embed_size=100, device=device)
elif model_type == "FE_F1":
    predictor = FE_F1(input_size, output_size, embed_size=100, device=device)
elif model_type == "FE_Dif":
    predictor = FE_Dif(input_size, output_size, embed_size=100, device=device)
elif model_type == "MLPOracle":
    predictor = MLPOracle(input_size, output_size, hidden_parameters, device=device)
elif model_type == "MLP":
    predictor = MLP(input_size, output_size, device=device)
elif model_type == "TRANSFORMER":
    predictor = Transformer(input_size, output_size, device=device)
else:
    raise ValueError(f"Unknown type '{model_type}'")

print(f"{type(predictor).__name__}: {predictor.num_params()/1e6:.2f}M parameters")
logdir = os.path.join(logdir, f"{type(predictor).__name__}")
logger = SummaryWriter(logdir)

# train them
for epoch in trange(epochs):
    # find a batch of size 50_000 instead of 500_000
    permutation = torch.randperm(train_xs.shape[1])
    train_xs_batch = train_xs[:, permutation[:batch_size], :]
    train_ys_batch = train_ys[:, permutation[:batch_size], :]

    # sort data into examples and training data based on permutations
    example_xs = train_xs_batch[:, :example_data_size, :].to(device)
    example_ys = train_ys_batch[:, :example_data_size, :].to(device)
    xs = train_xs_batch[:, example_data_size:, :].to(device)
    ys = train_ys_batch[:, example_data_size:, :].to(device)

    # train each predictor
    train_loss, norm = predictor.train(example_xs, example_ys, xs, ys)
    logger.add_scalar("train/loss", train_loss, epoch)
    if norm:
        logger.add_scalar("train/norm", norm, epoch)

    # need to free memory
    del example_xs, example_ys, xs, ys

    # now do the same thing but only for testing on ood data
    test_xs_batch = testing_xs[:, permutation[:batch_size], :]
    test_ys_batch = testing_ys[:, permutation[:batch_size], :]

    example_xs = test_xs_batch[:, :example_data_size, :].to(device)
    example_ys = test_ys_batch[:, :example_data_size, :].to(device)
    xs = test_xs_batch[:, example_data_size:, :].to(device)
    ys = test_ys_batch[:, example_data_size:, :].to(device)

    # test each predictor
    test_loss = predictor.test(example_xs, example_ys, xs, ys)
    logger.add_scalar("test/loss", test_loss, epoch)

    # free memory
    del example_xs, example_ys, xs, ys
predictor.save(logdir)