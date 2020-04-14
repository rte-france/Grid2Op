#!/bin/bash

export DQN_NAME=dqn-X.0.0.x
export DQN_DATA=~/data_grid2op/rte_case14_realistic

./inspect_action_space.py --path_data $DQN_DATA

rm -rf ./logs/$DQN_NAME
./train.py \
    --num_pre_steps 256 \
    --num_train_steps 131072 \
    --path_data $DQN_DATA \
    --name $DQN_NAME \
    --num_frames 4

rm -rf ./logs-$DQN_NAME
./eval.py \
    --path_data $DQN_DATA \
    --path_model ./$DQN_NAME.h5 \
    --path_logs ./logs-$DQN_NAME \
    --nb_episode 10 \
    --max_steps=8000

./inspect_graph.py --logdir ./logs-$DQN_NAME/
