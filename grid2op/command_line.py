#!/usr/bin/env python3

import warnings

from grid2op.main import main_cli as mainEntryPoint
from grid2op.Download.download import download_cli as downloadEntryPoint

def main():
    mainEntryPoint()

def download():
    downloadEntryPoint()

def replay():
    try:
        from grid2op.Episode.EpisodeReplay import replay_cli as replayEntryPoint
        replayEntryPoint()
    except:
        warn_msg = "\nEpisode replay is missing an optional dependency\n" \
                   "Please run pip3 install grid2op[optional]"
        warnings.warn(warn_msg)
