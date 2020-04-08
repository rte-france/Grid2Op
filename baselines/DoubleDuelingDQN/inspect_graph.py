#!/usr/bin/env python3

import argparse
import json
import tensorflow as tf
import numpy as np

from grid2op.MakeEnv import make2
from grid2op.Plot.PlotPlotly import PlotPlotly

from DoubleDuelingDQNAgent import DoubleDuelingDQNAgent

def cli():
    parser = argparse.ArgumentParser(description="Graph inspector")
    parser.add_argument("--path_data", required=True,
                        help="Path to dataset root directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    env = make2(args.path_data)
    env.reset()
    env.render()
    print ("Done ?")
    done = input()
