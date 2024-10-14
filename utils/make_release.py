# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# This script will update automatically create a new release
# - setup.py
# - grid2op/__init__.py
# - docs/conf.py
# - Dockerfile

import sys
import os
import argparse
import re
import subprocess
import time


def start_subprocess_print(li, sleepbefore=2, cwd=None):
    print("Will execute command after {}s: \n\t{}".format(sleepbefore, " ".join(li)))
    time.sleep(sleepbefore)
    subprocess.run(li, cwd=cwd)


def modify_and_push_docker(version,  # grid2op version
                           path,
                           templateDockerFile_to_use="templateDockerFile",
                           docker_versions=[],
                           docker_tags=[]):
    # Dockerfile
    template_dockerfile = os.path.join(path, "utils", templateDockerFile_to_use)
    dockerfile = os.path.join(path, "Dockerfile")
    with open(template_dockerfile, "r") as f:
        new_setup = f.read()
    new_setup = re.sub("__VERSION__",
                       "v{}".format(version),
                       new_setup)
    with open(dockerfile, "w") as f:
        f.write(new_setup)

    # Create new docker containers
    for vers_ in docker_versions:
        start_subprocess_print(
            ["docker", "build"] + docker_tags + ["-t", "{}/grid2op:{}".format(dockeruser, vers_), "."], cwd=path)
        start_subprocess_print(["docker", "push", "{}/grid2op:{}".format(dockeruser, vers_)], cwd=path)


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
        maj_, min_, minmin_, *post = version.split(".")
    except Exception as exc_:
        raise RuntimeError(
            "script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1). "
            "Please modify \"--version\" argument")

    regex_version = "[0-9]+\.[0-9]+\.[0-9]+(.post[0-9]+){0,1}(.rc[0-9]+){0,1}(.pre[0-9]+){0,1}(.dev[0-9]+){0,1}"
    # TODO use the official regex !
    # see https://semver.org/ and https://regex101.com/r/Ly7O1x/3/
    # regex_version = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    regex_version_with_str = f"['\"]{regex_version}['\"]"
    if re.match("^{}$".format(regex_version), version) is None:
        raise RuntimeError(
            "script \"update_version\": version should be formated as XX.YY.ZZ (eg 0.3.1) and not {}. "
            "Please modify \"--version\" argument".format(
                version))

    # TODO re.search(reg_, "0.0.4-rc1").group("prerelease") -> rc1 (if regex_version is the official one)
    if re.search(f".*(\\.|-)(rc|pre|dev)[0-9]+$", version) is not None:
        is_prerelease = True
        print("This is a pre release, docker will NOT be pushed, github tag will NOT be made")
        time.sleep(2)
    else:
        is_prerelease = False
        print("This is sandard release, docker will be pushed, github tag will be added")
        time.sleep(2)
        
    if True:
        # setup.py
        setup_path = os.path.join(path, "setup.py")
        grid2op_init = os.path.join(path, "grid2op", "__init__.py")
        with open(grid2op_init, "r") as f:
            old_init = f.read()
            
        if not os.path.exists(setup_path):
            raise RuntimeError(
                "script \"update_version\" cannot find the root path of Grid2op. "
                "Please provide a valid \"--path\" argument.")
        with open(setup_path, "r") as f:
            new_setup = f.read()
        try:
            old_version = re.search("__version__ = {}".format(regex_version_with_str), old_init).group(0)
        except Exception as e:
            raise RuntimeError("Impossible to find the old version number. Stopping here")
        
        old_version = re.sub("__version__ = ", "", old_version)
        old_version = re.sub("'", "", old_version)
        old_version = re.sub('"', "", old_version)
        old_version = re.sub("\\.rc[0-9]+", "", old_version)
        old_version = re.sub("\\.post[0-9]+", "", old_version)
        old_version = re.sub("\\.pre[0-9]+", "", old_version)
        old_version = re.sub("\\.dev[0-9]+", "", old_version)
        if version < old_version:
            raise RuntimeError("You provided the \"new\" version \"{}\" which is older (or equal) to the current version "
                               "found: \"{}\".".format(version, old_version))

        new_setup = re.sub("version={}".format(regex_version_with_str),
                           "version='{}'".format(version),
                           new_setup)
        
        with open(setup_path, "w") as f:
            f.write(new_setup)

        # Stage in git
        start_subprocess_print(["git", "add", setup_path])

        # grid2op/__init__.py
        with open(grid2op_init, "r") as f:
            new_setup = f.read()
        new_setup = re.sub("__version__ = {}".format(regex_version_with_str),
                           "__version__ = '{}'".format(version),
                           new_setup)     
        
        with open(grid2op_init, "w") as f:
            f.write(new_setup)
            
        # Stage in git
        start_subprocess_print(["git", "add", grid2op_init])

        # docs/conf.py
        docs_conf = os.path.join(path, "docs", "conf.py")
        with open(docs_conf, "r") as f:
            new_setup = f.read()
        new_setup = re.sub("release = {}".format(regex_version_with_str),
                           "release = '{}'".format(version),
                           new_setup)
        new_setup = re.sub("version = '[0-9]+\.[0-9]+'",
                           "version = '{}.{}'".format(maj_, min_),
                           new_setup)
        with open(docs_conf, "w") as f:
            f.write(new_setup)
        # Stage in git
        start_subprocess_print(["git", "add", docs_conf])

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

        if not is_prerelease:
            # Stage in git
            start_subprocess_print(["git", "add", dockerfile])

            # generate some logs, for backward compatibility
            # NB this generation is part of the test run, so it's safe to re generate the log when each version is released
            # in the sense that the tests pass ;-)
            import grid2op
            from grid2op.Agent import RandomAgent
            from grid2op.Runner import Runner
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                PATH_PREVIOUS_RUNNER = os.path.join(path, "grid2op", "data_test", "runner_data")
                # set the right grid2op version (instead of reloading the stuff, ugly, but working)
                grid2op.__version__ = version
                env = grid2op.make("rte_case5_example", test=True)
                runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)
                runner.run(nb_episode=2,
                        path_save=os.path.join(PATH_PREVIOUS_RUNNER, f"res_agent_{version}"),
                        pbar=True,
                        max_iter=100)
            # Stage in git
            start_subprocess_print(["git", "add", f'{os.path.join(PATH_PREVIOUS_RUNNER, f"res_agent_{version}")}/*'])

        # Commit
        start_subprocess_print(["git", "commit", "-S", "-m", "Release v{}".format(version)])
        if not is_prerelease:
            # Create a new git tag
            start_subprocess_print(["git", "tag", "-s", "-a", "v{}".format(version), "-m", "Release v{}".format(version)])

    if is_prerelease:
        print("Please push changes: 'git push'")
        sys.exit(0)
        
    # Wait for user to push changes
    pushed = input("Please push changes: 'git push && git push --tags' - then press any key")
    # TODO refacto these, no need to have 3 times almost the same "templatedockerfile"

    # update docker for test version
    # TODO remove the "-e" in this docker file, and copy paste the data in data_test in the appropriate folder
    # that you can get with a python call
    modify_and_push_docker(version, path=path,
                           templateDockerFile_to_use="templateDockerFile_test",
                           docker_versions=["test"],
                           docker_tags=["--no-cache"])
    # update docker for "light"
    modify_and_push_docker(version, path=path,
                           templateDockerFile_to_use="templateDockerFile_light",
                           docker_versions=[f"{version}-light"],
                           docker_tags=["--no-cache"])
    # update version for competition and regular version
    modify_and_push_docker(version,
                           path=path,
                           docker_versions=[version, "latest"],
                           docker_tags=["--no-cache"])
