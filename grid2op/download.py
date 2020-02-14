import argparse
import os
import sys

from .DownloadDataset import main_download, DEFAULT_PATH_DATA, LI_VALID_ENV

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download some datasets compatible with grid2op.')
    parser.add_argument('--path_save', default=DEFAULT_PATH_DATA, type=str,
                        help='The path where the data will be downloaded.')
    parser.add_argument('--name', default="case14_redisp", type=str,
                        help='The name of the dataset (one of {} ).'
                             ''.format(",".join(LI_VALID_ENV))
                        )

    args = parser.parse_args()
    dataset_name = args.name
    try:
        path_data = os.path.abspath(args.path_save)
    except Exception as e:
        print("Argument \"--path_save\" should be a valid path (directory) on your machine.")
        sys.exit(1)

    main_download(dataset_name, path_data)