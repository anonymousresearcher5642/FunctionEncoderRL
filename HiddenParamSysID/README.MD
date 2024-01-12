# Hidden Parameter System Identification

This code can learn a function encoder to 
predict system dynamics for a system with hidden parameters. Data from example
transitions are used to improve the prediction accuracy. See the paper for more details.

## Installation
```commandline
cd HiddenParamSysID # if you have not already
python -m venv venv
source venv/bin/activate
pip install torch stable-baselines3 tqdm tensorboard opencv-python numpy  matplotlib pandas
```


## Usage
The repo has the following structure
* A modified Half Cheetah environment where the segment lengths, control authory, and friction are randomized.
* 1.train_policies.py - Samples random dynamics, and then trains a policy via normal RL to walk forward. These policies are later used to gather data.
* 2.visualize_policies.py - Visualizes the policies trained in 1. This is useful to ensure the state-action space is being explored. 
* 3.gather_data.py - Uses the policies trained in 1. to gather data in numerous randomly sampled environments. This data is saved to disk for later training.
* 4.train_predictors.py - This file trains the various algorithms based on the data gathered in three. Algorithm code is in MultiSystemIdentification/
* 5.compute_encodings.py - This file is used to compute the reward encodings for a given hidden parameter dimension, used for the cosine similiarity plot. The representations are saved.
* 6.graph_cos_sim.py - This file is used to graph the cosine similarity.
* graph.py - This file takes the training data from tensorboard and writes it to csv, to be plotted in latex. It also creates a matplotlib plot.

Call './run_experiment.sh' to run all scripts in order. Be warned this will likely take many days to complete. 
