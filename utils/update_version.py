# This script will update automatically the different version number in the different files:
# - setup.py
# - grid2op/__init__.py
# - docs/conf.py

import os
import argparse
import re
import sys
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update the version of grid2op in the python files.')
    parser.add_argument('--version', default=None,
                        help='The new version to update.')
    parser.add_argument('--path', default=os.path.abspath("."),
                        help='The path of the root directory of Grid2op (default {}'.format(os.path.abspath(".")))
    args = parser.parse_args()
    path = args.path
    version = args.version

    if args.version is None:
        raise RuntimeError("script \"update_version\" should be called with a version number.")

    try:
        maj_, min_, minmin_ = version.split(".")
    except:
        raise RuntimeError("script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1). Please modify \"--version\" argument")

    if re.match('^[0-9]+\.[0-9]+\.[0-9]+$', version) is None:
        raise RuntimeError("script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1) and not {}. Please modify \"--version\" argument".format(version))

    # setup.py
    setup_path = os.path.join(path, "setup.py")
    if not os.path.exists(setup_path):
        raise RuntimeError("script \"update_version\" cannot find the root path of Grid2op. Please provide a valid \"--path\" argument.")
    with open(setup_path, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("version='[0-9]+\.[0-9]+\.[0-9]+'",
                       "version='{}'".format(version),
                       new_setup)
    with open(setup_path, "w") as f:
        f.write(new_setup)

    #grid2op/__init__.py
    grid2op_init = os.path.join(path, "grid2op", "__init__.py")
    with open(grid2op_init, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("__version__ = '[0-9]+\.[0-9]+\.[0-9]+'",
                       "__version__ = '{}'".format(version),
                       new_setup)
    with open(grid2op_init, "w") as f:
        f.write(new_setup)

    # docs/conf.py
    docs_conf = os.path.join(path, "docs", "conf.py")
    with open(docs_conf, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("release = '[0-9]+\.[0-9]+\.[0-9]+'",
                       "release = '{}'".format(version),
                       new_setup)
    new_setup = re.sub("version = '[0-9]+\.[0-9]+'",
                       "version = '{}.{}'".format(maj_, min_),
                       new_setup)
    with open(docs_conf, "w") as f:
        f.write(new_setup)