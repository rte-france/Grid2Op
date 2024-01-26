#!/usr/bin/env python3

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import os
import unittest
import sys

from grid2op.main import main_cli as mainEntryPoint
from grid2op.Download.download import main as downloadEntryPoint

__LI_FILENAME_TESTS = [
    "test_Action.py",
    # "test_Action_iadd.py",
    "test_ActionProperties.py",
    "test_Observation.py",
    "test_AgentsFast.py",
    "test_RunnerFast.py",
    "test_attached_envs.py",
    # "test_GymConverter.py",  # requires gym
    # "test_Reward.py",
    # "test_issue_126.py",
    # "test_issue_131.py",
    # "test_issue_140.py",
    # "test_issue_146.py",
    # "test_issue_147.py",
    # # "test_issue_148.py",  # requires additional data
    # "test_issue_151.py",
    # "test_issue_153.py",
    # "test_issue_164.py",
]


def main():
    mainEntryPoint()


def download():
    downloadEntryPoint()


def replay():
    try:
        from grid2op.Episode.EpisodeReplay import main as replayEntryPoint

        replayEntryPoint()
    except ImportError as e:
        warn_msg = (
            "\nEpisode replay is missing an optional dependency\n"
            "Please run pip3 install grid2op[optional].\n The error was {}"
        )
        warnings.warn(warn_msg.format(e))


def testinstall():
    """
    Performs basic tests to make sure grid2op is properly installed and working.

    It's not because these tests pass that grid2op will be fully functional however.
    """
    test_loader = unittest.TestLoader()
    this_directory = os.path.abspath(os.path.dirname(__file__))
    test_suite = test_loader.discover(
        os.path.join(this_directory, "tests"), pattern=__LI_FILENAME_TESTS[0]
    )
    for file_name in __LI_FILENAME_TESTS[1:]:
        test_suite.addTest(
            test_loader.discover(
                os.path.join(this_directory, "tests"), pattern=file_name
            )
        )
        
    def fun(first=None, *args, **kwargs):
        if first is not None:
            sys.stderr.write(first, *args, **kwargs)
        sys.stderr.write("\n")
    sys.stderr.writeln = fun
    results = unittest.TextTestResult(stream=sys.stderr,
                                      descriptions=True,
                                      verbosity=2)
    test_suite.run(results)
    if results.wasSuccessful():
        return 0
    else:
        print("\n")
        results.printErrors()
        # for _, str_ in results.errors:
        #     print(str_)
        #     print("-------------------------\n")
        # for _, str_ in results.failures:
        #     print(str_)
        #     print("-------------------------\n")
        raise RuntimeError("Test not successful !")
