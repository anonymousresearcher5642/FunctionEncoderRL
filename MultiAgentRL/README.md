# Multi-agent RL

This repo is code for a multi-agent RL with a function encoder.

## Installation
```commandline
# use some version of python. I have 3.10 but it should not matter too much.
cd MultiAgentRL # if you have not already
python -m venv venv
source venv/bin/activate
pip install gymnasium numpy torch pettingzoo tensorboard tianshou tqdm pandas matplotlib scipy opencv-python
```

## Usage
The code first trains a league of agents using normal PPO.

```python 1.test_league --epoch 1000```

Then it trains an encoder to predict each policy. The resulting function encoder
can represent any of their policies as a vector.

```python 2.train_encoder --resume-path "log/tag/league/2023-10-09 08:55:58" --embed-dim 100```

You will have to modify the above resume-path to fit your experiment.

Next, it trains a new player via various algorithms. Run ```./experiment.sh``` to train numerous algorithms.
Or, you can run ```3.train_new_player.py``` directly.
**You will have to modify the paths in experiment.sh also**

Lastly, you can plot using ```4.plot.py```.



