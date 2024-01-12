This repo is bootstrapped from 'Learning One Representation to Optimize All Rewards'

Thank you to the Ahmed Touati Et Al for proving the source code. Here is a link to their paper: 

[Learning One Representation to Optimize All Rewards.
Ahmed Touati, Yann Ollivier. NeurIPS 2021](https://arxiv.org/pdf/2103.07945.pdf)

## Install Requirements

```bash
# Some version of python. I use 3.9 for this project, but other versions may work.

# change to the directory and create a venv
cd MultiTaskRL # if you have not already
python -m venv venv
source venv/bin/activate

# install requirements
pip install numpy torch gym[atari] tqdm matplotlib opencv-python atari-py tensorflow pandas tensorboard
pip install git+https://github.com/mila-iqia/atari-representation-learning.git
pip install git+https://github.com/openai/baselines

# need to get atari roms for ms pacman
python get_roms.py
sudo apt-get install unrar
unrar x Roms.rar
mkdir rars
mv HC\ ROMS   rars
python -m atari_py.import_roms rars

```

## Instruction to run the code
All experiments can be run via ```./run_experiment.sh```. This will take a long time to complete. 

Then plot using ```python graph.py``` and ```python plot_atari_reward_encoder_image.py```
You will have to change the date-time path to fit your experiment. It will throw a error and show you where. 


## Note
This repo also contains code for a discrete and continuous grid world. This code is not used for the paper, but is kept in because there are dependencies. 