#!/usr/bin/env python3

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import warnings

from grid2op.main import main_cli as mainEntryPoint
from grid2op.Download.download import main as downloadEntryPoint


def main():
    mainEntryPoint()


def download():
    downloadEntryPoint()


def replay():
    try:
        from grid2op.Episode.EpisodeReplay import main as replayEntryPoint
        replayEntryPoint()
    except ImportError as e:
        warn_msg = "\nEpisode replay is missing an optional dependency\n" \
                   "Please run pip3 install grid2op[optional].\n The error was {}"
        warnings.warn(warn_msg.format(e))
