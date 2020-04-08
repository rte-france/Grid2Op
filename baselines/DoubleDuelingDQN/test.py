#!/usr/bin/env python3

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make2
from grid2op.Reward import RedispReward


def cli():
    parser = argparse.ArgumentParser(description="Test load all chronics")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    env = make2(args.path_data,
                reward_class=RedispReward)

    
    for i in range(10000):
        print ("Loading {}".format(i))
        env.reset()
        print ("OK - ", env.chronics_handler.get_id())
        d = False
        while d is False:
            obs, r, d, info = env.step(env.action_space({}))
        
