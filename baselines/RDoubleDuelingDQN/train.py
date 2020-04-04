#!/usr/bin/env python3

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make2
from grid2op.Reward import RedispReward

from RDoubleDuelingDQNAgent import RDoubleDuelingDQNAgent as RDQNAgent
from CustomAction import CustomAction

def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--name", required=True,
                        help="The name of the model")
    parser.add_argument("--batch_size", required=False,
                        default=32, type=int,
                        help="Mini batch size (defaults to 32)")
    parser.add_argument("--num_pre_steps", required=False,
                        default=256, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=1024, type=int,
                        help="Number of training iterations")
    parser.add_argument("--trace_len", required=False,
                        default=8, type=int,
                        help="Size of the trace to use during training")
    parser.add_argument("--learning_rate", required=False,
                        default=1e-5, type=float,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument("--resume", required=False,
                        help="Path to model.h5 to resume training with")
    return parser.parse_args()

if __name__ == "__main__":
    # Get params from command line
    args = cli()

    # Load grid2op game
    env = make2(args.path_data,
                reward_class=RedispReward,
                action_class=CustomAction)

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = RDQNAgent(env, env.action_space,
                      name=args.name, 
                      batch_size=args.batch_size,
                      trace_length=args.trace_len,
                      is_training=True,
                      lr=args.learning_rate)

    if args.resume is not None:
        agent.load_network(args.resume)

    agent.train(args.num_pre_steps, args.num_train_steps)
