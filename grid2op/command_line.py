#!/usr/bin/env python3

from grid2op.main import main_cli as mainEntryPoint
from grid2op.download import download_cli as downloadEntryPoint
from grid2op.Episode.EpisodeReplay import replay_cli as replayEntryPoint

def main():
    mainEntryPoint()

def download():
    downloadEntryPoint()

def replay():
    replayEntryPoint()
