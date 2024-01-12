#!/bin/bash

python 1.train_policies.py
python 2.visualize_policies.py
python 3.gather_data.py

python 4.train_predictors.py --model_type MLP --seed 0
python 4.train_predictors.py --model_type FE --seed 0
python 4.train_predictors.py --model_type MLPOracle --seed 0
python 4.train_predictors.py --model_type TRANSFORMER --seed 0
python 4.train_predictors.py --model_type FE_Dif --seed 0


python 4.train_predictors.py --model_type MLP --seed 1
python 4.train_predictors.py --model_type FE --seed 1
python 4.train_predictors.py --model_type MLPOracle --seed 1
python 4.train_predictors.py --model_type TRANSFORMER --seed 1
python 4.train_predictors.py --model_type FE_Dif --seed 1

python 4.train_predictors.py --model_type MLP --seed 2
python 4.train_predictors.py --model_type FE --seed 2
python 4.train_predictors.py --model_type MLPOracle --seed 2
python 4.train_predictors.py --model_type TRANSFORMER --seed 2
python 4.train_predictors.py --model_type FE_Dif --seed 2

python 5.compute_encodings.py --dimensions_to_investigate friction

python 5.compute_encodings.py --dimensions_to_investigate torso_length
python 5.compute_encodings.py --dimensions_to_investigate bthigh_length
python 5.compute_encodings.py --dimensions_to_investigate bshin_length
python 5.compute_encodings.py --dimensions_to_investigate bfoot_length
python 5.compute_encodings.py --dimensions_to_investigate fthigh_length
python 5.compute_encodings.py --dimensions_to_investigate fshin_length
python 5.compute_encodings.py --dimensions_to_investigate ffoot_length

python 5.compute_encodings.py --dimensions_to_investigate bthigh_gear
python 5.compute_encodings.py --dimensions_to_investigate bshin_gear
python 5.compute_encodings.py --dimensions_to_investigate bfoot_gear
python 5.compute_encodings.py --dimensions_to_investigate fthigh_gear
python 5.compute_encodings.py --dimensions_to_investigate fshin_gear
python 5.compute_encodings.py --dimensions_to_investigate ffoot_gear

python 6.graph_cos_sim.py