"""
TODO documentation of this function!

"""
import os
import pkg_resources
import argparse

from .Observation import ObservationHelper, CompleteObservation, ObsEnv

from .ChronicsHandler import Multifolder

from .Reward import FlatReward
from .Agent import DoNothingAgent
from .BackendPandaPower import PandaPowerBackend
from .GameRules import AllwaysLegal
from .Runner import Runner

DEFAULT_TEST_CASE = os.path.join(pkg_resources.resource_filename(__name__, 'data'),
                                 "test_PandaPower", "test_case14.json")

DEFAULT_CHRONICS_DATA = os.path.join(pkg_resources.resource_filename(__name__, 'data'),
                                     "test_multi_chronics")


def main(path_casefile=None,
         path_chronics=None,
         path_parameters=None,
         chronics_class=Multifolder,
         backend_class=PandaPowerBackend,
         agent_class=DoNothingAgent,
         reward_class=FlatReward,
         observation_class=CompleteObservation,
         legalAct_class=AllwaysLegal,
         nb_episode=3,
         nb_process=1,
         path_save=None,
         names_chronics_to_backend=None,
         gridStateclass_kwargs={}):
    if path_casefile is None:
        init_grid_path = DEFAULT_TEST_CASE
    else:
        init_grid_path = os.path.abspath(path_casefile)

    if path_chronics is None:
        path_chron = DEFAULT_CHRONICS_DATA
    else:
        path_chron = os.path.abspath(path_chronics)

    parameters_path = path_parameters

    runner = Runner(init_grid_path=init_grid_path,
                    path_chron=path_chron,
                    parameters_path=parameters_path,
                    names_chronics_to_backend=names_chronics_to_backend,
                    gridStateclass=chronics_class,
                    gridStateclass_kwargs=gridStateclass_kwargs,
                    backendClass=backend_class,
                    rewardClass=reward_class,
                    agentClass=agent_class,
                    observationClass=observation_class,
                    legalActClass=legalAct_class)

    res = runner.run(nb_episode=nb_episode, nb_process=nb_process, path_save=path_save)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--path_save', default=None,
                        help='The path where the log of the experience will be stored (default: None -> nothing stored)')
    parser.add_argument('--nb_process', type=int, default=1,
                        help='The number of process used for each evaluation (note that if nb_process > nb_episode then nb_episode is used.')
    parser.add_argument('--nb_episode', type=int, default=3,
                        help='The number of episode to play (default 3)')
    parser.add_argument('--path_casefile', type=str, default=DEFAULT_TEST_CASE,
                        help='Path where the case file is located (casefile is the file describing the powergrid)')
    parser.add_argument('--path_chronics', type=str, default=DEFAULT_CHRONICS_DATA,
                        help='Path where the chronics (temporal variation of loads and production usually are located)')
    parser.add_argument('--path_parameters', default=None,
                        help='Path where the _parameters of the game are stored')

    args = parser.parse_args()

    if args.path_save is not None:
        path_save = str(args.path_save)
    else:
        path_save = None
    if args.path_parameters is not None:
        path_parameter = str(args.path_parameters)
    else:
        path_parameter = None
    if args.path_casefile == DEFAULT_TEST_CASE and args.path_chronics == DEFAULT_CHRONICS_DATA:
        names_chronics_to_backend = {"loads": {"2_C-10.61": 'load_1_0', "3_C151.15": 'load_2_1',
                                               "14_C63.6": 'load_13_2', "4_C-9.47": 'load_3_3',
                                               "5_C201.84": 'load_4_4',
                                               "6_C-6.27": 'load_5_5', "9_C130.49": 'load_8_6',
                                               "10_C228.66": 'load_9_7',
                                               "11_C-138.89": 'load_10_8', "12_C-27.88": 'load_11_9',
                                               "13_C-13.33": 'load_12_10'},
                                     "lines": {'1_2_1': '0_1_0', '1_5_2': '0_4_1', '9_10_16': '8_9_2',
                                               '9_14_17': '8_13_3',
                                               '10_11_18': '9_10_4', '12_13_19': '11_12_5', '13_14_20': '12_13_6',
                                               '2_3_3': '1_2_7', '2_4_4': '1_3_8', '2_5_5': '1_4_9',
                                               '3_4_6': '2_3_10',
                                               '4_5_7': '3_4_11', '6_11_11': '5_10_12', '6_12_12': '5_11_13',
                                               '6_13_13': '5_12_14', '4_7_8': '3_6_15', '4_9_9': '3_8_16',
                                               '5_6_10': '4_5_17',
                                               '7_8_14': '6_7_18', '7_9_15': '6_8_19'},
                                     "prods": {"1_G137.1": 'gen_0_4', "3_G36.31": "gen_2_1", "6_G63.29": "gen_5_2",
                                               "2_G-56.47": "gen_1_0", "8_G40.43": "gen_7_3"},
                                     }
    else:
        names_chronics_to_backend = None

    # actually performing the run
    msg_ = "Running Grid2Op:\n\t- on case file at \"{case_file}\"\n\t- with data located at \"{data}\""
    msg_ += "\n\t- using {process} process(es)\n\t- for {nb_episode} episodes"
    if args.path_save is None:
        msg_ += "\n\t- results will not be saved"
    else:
        msg_ += "\n\t- results will be saved in \"{}\"".format(args.path_save)
    print(msg_.format(case_file=args.path_casefile, data=args.path_chronics, process=args.nb_process,
                      nb_episode=args.nb_episode))
    res = main(path_save=path_save,
               nb_process=args.nb_process,
               nb_episode=args.nb_episode,
               path_casefile=args.path_casefile,
               path_chronics=args.path_chronics,
               path_parameters=path_parameter,
               names_chronics_to_backend=names_chronics_to_backend
               )
    print("The results are:")
    for chron_name, _, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.2f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)