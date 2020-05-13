#!/usr/bin/env python3

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
