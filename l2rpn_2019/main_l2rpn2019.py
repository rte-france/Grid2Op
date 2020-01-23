import argparse
import os

from create_env import main, PATH_DATA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--path_save', default=None,
                        help='The path where the log of the experience will be stored '
                             '(default: None -> nothing stored)')
    parser.add_argument('--nb_process', type=int, default=1,
                        help='The number of process used for each evaluation (note that if nb_process > '
                             'nb_episode then nb_episode is used.')
    parser.add_argument('--nb_episode', type=int, default=1,
                        help='The number of episode to play (default 1)')
    parser.add_argument('--path_chronics', type=str, default=PATH_DATA,
                        help='Path where the chronics (temporal variation of loads and production usually) are located')
    parser.add_argument('--path_parameters', default=None,
                        help='Path where the parameters of the game are stored')

    args = parser.parse_args()

    if args.path_save is not None:
        path_save = str(args.path_save)
    else:
        path_save = None

    if args.path_parameters is not None:
        path_parameter = str(args.path_parameters)
    else:
        path_parameter = None

    if not os.path.exists(args.path_chronics):
        # todo complete that
        raise RuntimeError("Unable to find L2RPN 2019 chronics at \"{}\". You can download the training"
                           "set with \"python utils\\\"".format(args.path_chronics))

    res = main(path_save=path_save,
               submission_dir=".",
               nb_process=args.nb_process,
               nb_episode=args.nb_episode,
               path_chronics=args.path_chronics,
               path_parameters=path_parameter)
