# This files allows to automatically update the documentation of grid2op on the
# readthedocs.io website. It should not be used for other purpose.

import argparse
import json
import os
import re
import subprocess

try:
    import requests as rq
except:
    raise RuntimeError("Impossible to find library urllib. Please install it.")

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
        raise RuntimeError("script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1)."
                           " Please modify \"--version\" argument")

    if re.match('^[0-9]+\.[0-9]+\.[0-9]+$', version) is None:
        raise RuntimeError("script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1) and not {}."
                           " Please modify \"--version\" argument".format(version))

    # update Dockerfile
    template_dockerfile = os.path.join(path, "utils", "templateDockerFile")
    dockerfile = os.path.join(path, "Dockerfile")
    with open(template_dockerfile, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("__VERSION__",
                       "v{}".format(version),
                       new_setup)
    with open(dockerfile, "w") as f:
        f.write(new_setup)

    # # push new version to dockerhub
    # for vers_ in [version, "latest"]:
    #     subprocess.run(["docker", "build", "-t", "bdonnot/grid2op:{}".format(version), "."], cwd=path)
    #     subprocess.run(["docker", "push", "bdonnot/grid2op:{}".format(version), "."], cwd=path)