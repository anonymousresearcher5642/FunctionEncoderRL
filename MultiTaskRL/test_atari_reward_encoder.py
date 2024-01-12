import numpy
import torch
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

from atari_modules.models import TaskAwareCritic, RewardEncoder, RewardEncoderTranslator
from matplotlib import cbook, cm

def get_reward_function(g, num_actions):
    g = torch.tensor(g, device="cpu") / 170.

    def reward_function(achieved_goal):
        distances = torch.sum((achieved_goal - g) ** 2, dim=1) ** 0.5
        # rewards = torch.where(distances < 0.07, torch.tensor([1.0], device=obs.device), torch.tensor([0.0], device=obs.device))
        rewards = -distances
        rewards = rewards.unsqueeze(1).expand(rewards.shape[0], num_actions).to(torch.float32)
        return rewards
        # g_canonical = featurizer.transform(g[None, :])
        # distances = torch.sum((obs - g_canonical) ** 2, dim=1) ** 0.5
        # # rewards = torch.where(distances < 0.07, torch.tensor([0.0], device=obs.device), torch.tensor([-1.0], device=obs.device))
        # rewards = -distances
        # rewards = rewards.unsqueeze(1).expand(rewards.shape[0], num_actions).to(torch.float32)
        # return rewards


    return reward_function









env_params = {'obs': (1,1),
          'goal': 2,
          'action': 5,
          'max_timesteps': 50,
          }
embed_dim=100

# create the networks
reward_encoder = RewardEncoder(env_params, embed_dim) #.to("cuda:0")
save_dir_date_time = '2023-09-28 10:53:03'
reward_encoder.load_state_dict(torch.load(f'data/{save_dir_date_time}/reward_encoder.pt', map_location="cpu"))

all_goals =  [[11, 9], [18, 9], [27, 9], [34, 9], [50, 9], [58, 9], [66, 9], [76, 9], [86, 9], [95, 9], [104, 9], [112, 9], [127, 9], [135, 9], [143, 9], [150, 9],
          [36, 21], [50, 21], [112, 21], [128, 21], [152, 21],
          [11, 34], [18, 34], [35, 34], [43, 34], [51, 34], [59, 34], [68, 34], [76, 34], [87, 34], [95, 34], [103, 34], [111, 34], [119, 34], [127, 34], [135, 34], [143, 34], [151, 34],
          [20, 46], [34, 46], [60, 46], [103, 46], [129, 46], [144, 46],
          [11, 57], [19, 57], [35, 57], [43, 57], [51, 57], [59, 57], [67, 57], [75, 57], [87, 57], [95, 57], [103, 57], [111, 57], [118, 57], [127, 57], [143, 57], [151, 57],
          [20, 69], [59, 69], [104, 69], [144, 69],
          [20, 82], [27, 82], [35, 82], [43, 82], [43, 82], [51, 82], [59, 82], [103, 82], [111, 82], [128, 82], [135, 82], [142, 82],
          [20, 93], [58, 93], [103, 93], [143, 93],
          [12, 105], [19, 105], [35, 105], [43, 105], [51, 105], [110, 105], [118, 105], [126, 105], [143, 105], [151, 105],
          [19, 117], [35, 117], [50, 117], [50, 117], [67, 117], [94, 117], [110, 117], [127, 117], [143, 117],
          [12, 129], [19, 129], [27, 129], [35, 129], [35, 129], [51, 129], [67, 129], [75, 129], [86, 129], [95, 129], [111, 129], [126, 129], [135, 129], [143, 129], [151, 129],
          [12, 141], [35, 141], [50, 141], [67, 141], [95, 141], [102, 141], [112, 141], [127, 141], [151, 141],
          [12, 153], [35, 153], [67, 153], [95, 153], [127, 153], [151, 153],
          [12, 165], [19, 165], [27, 165], [35, 165], [43, 165], [50, 165], [59, 165], [68, 165], [76, 165], [87, 165], [87, 165], [96, 165], [103, 165],
          [112, 165], [119, 165], [127, 165], [136, 165], [142, 165], [151, 165]]
print(len(all_goals))

x_min, y_min, x_max,y_max = 10000, 10000, -10000, -10000
for goal in all_goals:
    x_min = min(goal[0], x_min)
    y_min = min(goal[1], y_min)
    x_max = max(goal[0], x_max)
    y_max = max(goal[1], y_max)

# create grid over xs and ys
xs = torch.linspace(x_min, x_max, 100)# .to("cuda:0")
ys = torch.linspace(y_min, y_max, 100)# .to("cuda:0")
xs, ys = torch.meshgrid(xs, ys)
print(xs.shape, ys.shape)
xs_flat = xs.reshape(-1)
ys_flat = ys.reshape(-1)
print(xs_flat.shape, ys_flat.shape)
inputs = torch.stack([xs_flat, ys_flat], dim=1)# .to(torch.float32)# .to("cuda:0")
inputs = inputs / 170.

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

all_encoings = []
# plot goals and their predictions
for goal in all_goals:
    reward_function = get_reward_function(goal, 5)
    true_rewards = reward_function(inputs)

    # predict rewards
    individual_encodings = reward_encoder(inputs)
    encoding = torch.mean(true_rewards.unsqueeze(1) * individual_encodings, dim=0)
    all_encoings.append(encoding.cpu().detach())
    # normalization loss to satisfy definition of a basis
    estimated_reward = torch.sum(encoding * individual_encodings, dim=1)
    reward_prediction_loss = torch.mean((estimated_reward - true_rewards) ** 2)
    print(f"Goal {goal} reward prediction loss: {reward_prediction_loss.item():0.3f}")

    # plot
    # plot a 3d graph
    # Set up plot
    # xs_plot = xs.cpu().detach().numpy()
    # ys_plot = ys.cpu().detach().numpy()
    # zs_plot = estimated_reward[:,0].reshape(100, 100)
    # zs_plot = zs_plot.cpu().detach().numpy()



    # Create the 3D surface plot
    # ax.clear()
    #
    # surface = ax.plot_surface(xs_plot, ys_plot, zs_plot, cmap='viridis')  # You can choose a different colormap
    # # add a marker at goal location (x,y,z=0)
    # ax.scatter(goal[0], goal[1], 0, marker='o', color='red')
    #
    # # Add labels and a colorbar
    # # fig.colorbar(surface)
    # plt.show(block=False)
    # plt.pause(1)

for goal_to_sim in range(len(all_goals)):
    # goal_to_sim = 120
    sims = []
    for encoding in all_encoings:
        # compute cos similiarity
        sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
        # print(f"{sim.item():0.3f}")
        # print(encoding.flatten(), all_encoings[goal_to_sim].flatten())
        # print("\n\n")
        sims.append(sim.item())

    # Create the 3D surface plot
    ax.clear()
    xys = numpy.array(all_goals)
    # print(xys.shape, numpy.array(sims).shape)
    surface = ax.plot_trisurf(xys[:, 0], xys[:, 1], numpy.array(sims), cmap='viridis')  # You can choose a different colormap
    # add a marker at goal location (x,y,z=0)
    ax.scatter(all_goals[goal_to_sim][0], all_goals[goal_to_sim][1], 1, marker='o', color='red')

    # Add labels and a colorbar
    # fig.colorbar(surface)
    plt.show(block=False)
    plt.pause(1)