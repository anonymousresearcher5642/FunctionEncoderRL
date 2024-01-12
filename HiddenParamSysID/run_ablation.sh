#!/bin/bash
python 4.train_predictors.py --model_type FE --seed 0
python 4.train_predictors.py --model_type FE_PWN --seed 0
python 4.train_predictors.py --model_type FE_F1 --seed 0
python 4.train_predictors.py --model_type FE_Dif --seed 0
