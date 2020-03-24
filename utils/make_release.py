# This script will update automatically create a new release
# - setup.py
# - grid2op/__init__.py
# - docs/conf.py
# - Dockerfile

import os
import argparse
import re
import sys
import pdb
import json
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update the version of grid2op in the python files.')
    parser.add_argument('--version', default=None,
                        help='The new version to update.')
    parser.add_argument('--dockeruser', default='bdonnot',
                        help='The docker hub username.')
    parser.add_argument('--path', default=os.path.abspath("."),
                        help='The path of the root directory of Grid2op (default {}'.format(os.path.abspath(".")))
    args = parser.parse_args()
    path = args.path
    dockeruser = args.dockeruser
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
    # Stage in git
    subprocess.run(["git", "add", setup_path])

    #grid2op/__init__.py
    grid2op_init = os.path.join(path, "grid2op", "__init__.py")
    with open(grid2op_init, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("__version__ = '[0-9]+\.[0-9]+\.[0-9]+'",
                       "__version__ = '{}'".format(version),
                       new_setup)
    with open(grid2op_init, "w") as f:
        f.write(new_setup)
    # Stage in git
    subprocess.run(["git", "add", grid2op_init])
    
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
    # Stage in git
    subprocess.run(["git", "add", docs_conf])

    # Dockerfile
    template_dockerfile = os.path.join(path, "utils", "templateDockerFile")
    dockerfile = os.path.join(path, "Dockerfile")
    with open(template_dockerfile, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("__VERSION__",
                       "v{}".format(version),
                       new_setup)
    with open(dockerfile, "w") as f:
        f.write(new_setup)
    # Stage in git
    subprocess.run(["git", "add", dockerfile])
        
    # Commit
    os.path.expanduser("~")
    subprocess.run(["git", "commit", "-m", "Release v{}".format(version)])
    # Create a new git tag
    subprocess.run(["git", "tag", "-a", "v{}".format(version), "-m", "Release v{}".format(version)])

    # Wait for user to push changes
    pushed = input("Please push changes: 'git push && git push --tags' - then press any key")
    
    # Create new docker containers
    for vers_ in [version, "latest"]:
        subprocess.run(["docker", "build", "-t", "{}/grid2op:{}".format(dockeruser, vers_), "."], cwd=path)
        subprocess.run(["docker", "push", "{}/grid2op:{}".format(dockeruser, vers_)], cwd=path)
    
