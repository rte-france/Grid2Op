#!/usr/bin/env python3

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make2
from grid2op.Reward import *
from grid2op.Action import *

from DoubleDuelingDQNAgent import DoubleDuelingDQNAgent as DDDQNAgent


def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--name", required=True,
                        help="The name of the model")
    parser.add_argument("--batch_size", required=False,
                        default=32, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--num_pre_steps", required=False,
                        default=256, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=1024, type=int,
                        help="Number of training iterations")
    parser.add_argument("--num_frames", required=False,
                        default=4, type=int,
                        help="Number of observation frames to use during training")
    parser.add_argument("--learning_rate", required=False,
                        default=1e-5, type=float,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument("--resume", required=False,
                        help="Path to model.h5 to resume training with")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    # Create grid2op game environement
    env = make2(args.path_data,
                action_class=PowerlineChangeAndDispatchAction,
                reward_class=CombinedReward,
                other_rewards={
                    "bridge": BridgeReward,
                    "overflow": CloseToOverflowReward,
                    "distance": DistanceReward
                })
    # Register custom reward for training
    cr = env.reward_helper.template_reward
    cr.addReward("bridge", BridgeReward(), 0.33)
    cr.addReward("overflow", CloseToOverflowReward(), 0.33)
    cr.addReward("distance", DistanceReward(), 0.33)
    cr.addReward("game", GameplayReward(), 1.0)
    #cr.addReward("redisp", RedispReward(), 2.5e-4)
    # Initialize custom rewards
    cr.initialize(env)

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = DDDQNAgent(env, env.action_space,
                       name=args.name, 
                       is_training=True,
                       batch_size=args.batch_size,
                       num_frames=args.num_frames,
                       lr=args.learning_rate)

    if args.resume is not None:
        agent.load_network(args.resume)

    agent.train(args.num_pre_steps, args.num_train_steps)
