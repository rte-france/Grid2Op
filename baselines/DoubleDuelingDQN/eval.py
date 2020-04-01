#!/usr/bin/env python

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make2
from grid2op.Runner import Runner

from DoubleDuelingDQNAgent import DoubleDuelingDQNAgent as DDDQNAgent
from CustomEconomicReward import CustomEconomicReward

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
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create dataset env
    env = make2(args.path_data, reward_class=CustomEconomicReward)

    # Instanciate agent
    agent = DDDQNAgent(env.action_space, num_frames=4)
    # Get shapes from env
    obs = env.reset()
    state = agent.convert_obs(obs)
    # Build Model
    agent.init_deep_q(state)
    # Load weights from file
    agent.deep_q.load_network(args.path_model)

    # Build runner
    runner_params = env.get_params_for_runner()
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Run
    res = runner.run(path_save=args.path_logs,
                     nb_episode=args.nb_episode,
                     max_iter=1000, pbar=True)
    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

