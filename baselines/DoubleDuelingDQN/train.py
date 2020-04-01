#!/usr/bin/env python3

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make2
from grid2op.Reward import RedispReward

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
    env = make2(args.path_data, reward_class=RedispReward)

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
