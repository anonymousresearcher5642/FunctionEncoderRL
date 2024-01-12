from datetime import datetime

import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# seed torch
torch.manual_seed(0)

# hyper params
k = 100
n = 1 # f: R^n -> R
X = [-10, 10]
batches = 1000
fs_per_batch = 10
samples = 10000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
f_type = "P_2" # "P_2" or "sets"
total_samples = fs_per_batch * samples
min_samples = 100


# create logger
current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
dir = f"./test_data/" + date_time_string
logger = SummaryWriter(dir)

# computes the encoding for a function encoder given the function f and the inputs xs.
def get_encoding(f, xs, encoder):
    ys = f(xs)
    individual_encodings = encoder(xs.reshape(-1, 1))
    return torch.mean(ys.unsqueeze(2) * individual_encodings.unsqueeze(0), dim=1)


# this is used to generate artifical data.
def sample_function(type, num_functions=1, get_description=False, out_of_distribution=False):
    if type == "P_2":
        if out_of_distribution:
            low, high = [-100000.0, -100000.0, -100000.0], [100000.0, 100000.0, 100000.0]
        else:
            low, high = [-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]
        a = torch.rand(num_functions, 1, device=device) * (high[0] - low[0]) + low[0]
        b = torch.rand(num_functions, 1, device=device) * (high[1] - low[1]) + low[1]
        c = torch.rand(num_functions, 1, device=device) * (high[2] - low[2]) + low[2]

        f = lambda x: a * x**2 + b * x + c
        if get_description:
            return f, f"{a.item():0.2f}x^2 + {b.item():0.2f}x + {c.item():0.2f}"
        else:
            return f
    elif type == "sets":
        # locations = [torch.tensor(-5.0, device=device), torch.tensor(5.0, device=device)]
        # location = locations[0] if torch.rand(1) < 0.5 else locations[1]
        if out_of_distribution:
            locations = torch.rand(3, device=device) * (X[1] - X[0]) + X[0]
            scale = torch.rand(1, device=device) * (5- -5) + -5
            f = lambda x: scale * (torch.where(torch.abs(x - locations[0]) < 1.0, torch.ones_like(x), torch.zeros_like(x)) +
                                   torch.where(torch.abs(x - locations[1]) < 1.0, torch.ones_like(x), torch.zeros_like(x)) +
                                   torch.where(torch.abs(x - locations[2]) < 1.0, torch.ones_like(x), torch.zeros_like(x)))

            if get_description:
                return f, f"Locs: {locations[0].item():0.2f},{locations[1].item():0.2f},{locations[2].item():0.2f},"
            else:
                return f
        else:
            location = torch.rand(1, device=device) * (X[1] - X[0]) + X[0]
            f = lambda x: torch.where(torch.abs(x - location) < 1.0, torch.ones_like(x), torch.zeros_like(x))
            if get_description:
                return f, f"Loc: {location.item():0.2f}, Width: 1.0"
            else:
                return f
    else:
        raise ValueError(f"Unknown type {type}")

# samples from the input space uniformly
def sample_from_X(n):
    return (torch.rand(1, n, device=device) * (X[1] - X[0]) + X[0]).reshape(1, -1)

# learn g
encoder = torch.nn.Sequential(
    torch.nn.Linear(n, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, k),
).to(device)
optim = torch.optim.Adam(encoder.parameters(), lr=0.001)

print("Training data is saved via tensorboard.")
print("To view it, run: tensorboard --logdir ./test_data")
print("Then open http://localhost:6006/ in your browser")
print("After training, you can view the results in ./test_data/<date-time>/graph.png and ./test_data/<date-time>/graph_ood.png")

for i in trange(batches):
    # sample a function from F
    f = sample_function(f_type, num_functions=fs_per_batch)


    # compute the encoding of f
    xs = sample_from_X(int(samples))
    encoding = get_encoding(f, xs, encoder)
    assert encoding.shape == (fs_per_batch, k,), f"Encoding is wrong shape, got {encoding.shape}, expected {(k,)}"

    # compute approximations of F(x)
    ys = f(xs)
    y_hats = torch.sum(encoding.unsqueeze(1) * encoder(xs.reshape(-1, 1)).unsqueeze(0), dim=2)
    assert ys.shape == y_hats.shape, f"y_hats is wrong shape, got {y_hats.shape}, expected {ys.shape}"

    # compute prediction loss
    prediction_loss = torch.mean((ys - y_hats)**2)


    loss = prediction_loss
    loss.backward() # accumulates gradients

    # clip grads
    norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)

    with torch.no_grad():
        logger.add_scalar("loss", loss.item(), i)
        logger.add_scalar("grad_norm", norm.item(), i)

    # finally step
    optim.step()
    optim.zero_grad()

################################################################################################
############ The below code is only graphing, you can safely ignore it. ########################
################################################################################################



# test this thing in distribution.
with torch.no_grad():
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for i in range(3):
        for j in range(3):
            # compute an encoding
            f, description = sample_function(f_type, get_description=True, out_of_distribution=False)
            xs = sample_from_X(int(samples))
            encoding = get_encoding(f, xs, encoder)

            # graph y vs yhat
            linspace_xs = torch.linspace(X[0], X[1], 1000, device=device).reshape(-1, 1)
            ys = f(linspace_xs)
            y_hats = torch.sum(encoding * encoder(linspace_xs), dim=1).reshape(-1, 1)
            ax = axes[i, j]
            ax.plot(linspace_xs.cpu(), ys.cpu(), label=f'y')
            ax.plot(linspace_xs.cpu(), y_hats.cpu(), label=f'y_hat')
            ax.set_title(description)
            if i == 0 and j == 0:
                ax.legend()


            # print(f"Function: {description}")
            # print(f"Encoding: {encoding.detach()}")
            # for x,y,y_hat in zip(examles_xs, ys, y_hats):
            #     print(f"\tInput: {x.item():0.3f}, Output: {y.item():0.3f}, Predicted: {y_hat.item():0.3f}, Error = {abs(y.item() - y_hat.item()):0.3f}")
    plt.tight_layout()
    plt.savefig(f"{dir}/graph.png")

    # test this thing out of distribution
    plt.clf()
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for i in range(3):
        for j in range(3):
            # compute an encoding
            f, description = sample_function(f_type, get_description=True, out_of_distribution=True)
            xs = sample_from_X(int(samples))
            encoding = get_encoding(f, xs, encoder)

            # graph y vs yhat
            linspace_xs = torch.linspace(X[0], X[1], 1000, device=device).reshape(-1, 1)
            ys = f(linspace_xs)
            y_hats = torch.sum(encoding * encoder(linspace_xs), dim=1).reshape(-1, 1)
            ax = axes[i, j]
            ax.plot(linspace_xs.cpu(), ys.cpu(), label=f'y')
            ax.plot(linspace_xs.cpu(), y_hats.cpu(), label=f'y_hat')
            ax.set_title(description)
            if i == 0 and j == 0:
                ax.legend()


            # print(f"Function: {description}")
            # print(f"Encoding: {encoding.detach()}")
            # for x,y,y_hat in zip(examles_xs, ys, y_hats):
            #     print(f"\tInput: {x.item():0.3f}, Output: {y.item():0.3f}, Predicted: {y_hat.item():0.3f}, Error = {abs(y.item() - y_hat.item()):0.3f}")
    plt.tight_layout()
    plt.savefig(f"{dir}/graph_ood.png")