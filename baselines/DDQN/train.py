#!/usr/bin/env python

import argparse

from grid2op.MakeEnv import make2
from grid2op.Reward import RedispReward

from DDQNAgent import DDQNAgent 
from TrainAgent import TrainAgent

def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--name", required=True,
                        help="The name of the model")
    parser.add_argument("--batch_size", required=False,
                        default=1, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--num_pre_steps", required=False,
                        default=100, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=16, type=int,
                        help="Number of training iterations")
    parser.add_argument("--num_frames", required=False,
                        default=1, type=int,
                        help="Number of observation frames to use during training")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    env = make2(args.path_data)
    dqnn_agent = DDQNAgent(env.action_space, num_frames=args.num_frames)
    trainer = TrainAgent(dqnn_agent, env,
                         name=args.name, 
                         reward_fun=RedispReward,
                         num_frames=args.num_frames)
    trainer.train(args.num_pre_steps, args.num_train_steps)
