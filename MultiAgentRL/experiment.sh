#!/bin/bash

python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 1 --alg-type "FE_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 2 --alg-type "FE_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 3 --alg-type "FE_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 4 --alg-type "FE_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 5 --alg-type "FE_PPO"

python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 1 --alg-type "PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 2 --alg-type "PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 3 --alg-type "PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 4 --alg-type "PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 5 --alg-type "PPO"

python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 1 --alg-type "Transformer_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 2 --alg-type "Transformer_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 3 --alg-type "Transformer_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 4 --alg-type "Transformer_PPO"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 5 --alg-type "Transformer_PPO"


python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 1 --alg-type "Oracle_PPO" --device "cpu"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 2 --alg-type "Oracle_PPO" --device "cpu"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 3 --alg-type "Oracle_PPO" --device "cpu"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 4 --alg-type "Oracle_PPO" --device "cpu"
python 3.train_new_player.py --epoch 500 --encoder-dir "log/tag/encoder/2023-10-10 10:17:15" --resume-path "log/tag/league/2023-10-09 08:55:58" --seed 5 --alg-type "Oracle_PPO" --device "cpu"