#!/usr/bin/env python3

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make2
from grid2op.Runner import Runner
from grid2op.Reward import RedispReward

from RDoubleDuelingDQNAgent import RDoubleDuelingDQNAgent as RDQNAgent
from CustomAction import CustomAction

def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--path_model", required=True,
                        help="The path to the model [.h5]")
    parser.add_argument("--path_logs", required=False,
                        default="./logs_eval", type=str,
                        help="Path to output logs directory") 
    parser.add_argument("--nb_episode", required=False,
                        default=1, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=1, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=1000, type=int,
                        help="Maximum number of steps per scenario")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create dataset env
    env = make2(args.path_data, reward_class=RedispReward, action_class=CustomAction)

    # Create agent
    agent = RDQNAgent(env, env.action_space, is_training=False)
    # Load weights from file
    agent.load_network(args.path_model)

    # Build runner
    runner_params = env.get_params_for_runner()
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Run
    res = runner.run(path_save=args.path_logs,
                     nb_episode=args.nb_episode,
                     nb_process=args.nb_process,
                     max_iter=args.max_steps,
                     pbar=True)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

