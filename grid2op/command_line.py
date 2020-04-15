#!/usr/bin/env python3

from grid2op.main import main_cli as mainEntryPoint
from grid2op.download import download_cli as downloadEntryPoint

def main():
    mainEntryPoint()

def download():
    downloadEntryPoint()
