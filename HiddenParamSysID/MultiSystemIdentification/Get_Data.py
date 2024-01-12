import os
import numpy as np
import torch

def get_system_data(device):
    eps = 1e-8

    # loads state, action, next_state data from a directory. Returns xs, ys
    def load_data(dir):
        # load into numpy
        states = np.load(os.path.join(dir, "states.npy"))
        actions = np.load(os.path.join(dir, "actions.npy"))
        next_states = np.load(os.path.join(dir, "next_states.npy"))

        # convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # combine states and actions
        # note we do not have enough memory to put all of this data on GPU, so we will have to do it in batches
        xs = torch.cat((states, actions), dim=-1) # .to(device)
        ys = next_states  # .to(device)
        return xs, ys

    # dirs
    data_dir = "data/"

    # load all data
    xs, ys = load_data(data_dir)

    # normalize data since some dimensions are huge compared to others. This skews the loss functions. We want all dimensions to be on the same scale
    # we can leave action dims as is because they are all in range -1,1
    # but normalize observation spaces.

    # compute means and stds
    mean = torch.mean(xs[:, :, :17], dim=(0, 1))
    stds = torch.std(xs[:, :, :17], dim=(0, 1))
    assert mean.shape == (17,), f"mean is wrong shape, got {mean.shape}, expected {(17,)}"
    assert stds.shape == (17,), f"stds is wrong shape, got {stds.shape}, expected {(17,)}"

    # normalize all data
    xs[:, :, :17] = (xs[:, :, :17] - mean) / (stds + eps)
    ys = (ys - mean) / (stds + eps)

    # get sizes
    input_size = xs.shape[-1]
    output_size = ys.shape[-1]

    # split data into training data and testing data
    # note the environments are randomly generated originally, so we can simply cut it in half
    # while maintaining randomness
    train_xs = xs[:xs.shape[0]//2, :, :]
    train_ys = ys[:ys.shape[0]//2, :, :]
    test_xs = xs[xs.shape[0]//2:, :, :]
    test_ys = ys[ys.shape[0]//2:, :, :]

    return (train_xs, train_ys), (test_xs, test_ys), input_size, output_size


def get_polynomial_data(device):
    # test data to validate training process
    input_size = 1
    output_size = 1
    x_range = [-3, 3]
    poly_range = [-3, 3]
    number_polynomials = 200
    number_data_points = 50_000

    # generate polynomial coefficients
    poly_coeffs = torch.rand(number_polynomials, 3, device=device) * (poly_range[1] - poly_range[0]) + poly_range[0]

    # sample xs
    xs = torch.rand(number_data_points, input_size, device=device) * (x_range[1] - x_range[0]) + x_range[0]

    # for each quadratic, generate ys into a single tensor
    ys = torch.zeros(number_polynomials, number_data_points, output_size, device=device)
    for i in range(number_polynomials):
        ys[i, :, :] = poly_coeffs[i, 0] * xs**2 + poly_coeffs[i, 1] * xs + poly_coeffs[i, 2]

    # repeat xs for each polynomial
    xs = xs.repeat(number_polynomials, 1, 1)

    # break into train and test set
    train_xs = xs[:xs.shape[0]//2, :, :]
    train_ys = ys[:xs.shape[0]//2, :, :]
    test_xs = xs[xs.shape[0]//2:, :, :]
    test_ys = ys[xs.shape[0]//2:, :, :]
    return (train_xs, train_ys), (test_xs, test_ys), input_size, output_size

