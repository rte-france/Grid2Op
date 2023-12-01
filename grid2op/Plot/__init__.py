__all__ = [
    "BasePlot",
    "PlotMatplotlib",
    "PlotPlotly",
    "PlotPyGame",
    "Plotting",
    # "EpisodeReplay",
]

from grid2op.Plot.BasePlot import BasePlot
from grid2op.Plot.PlotMatplotlib import PlotMatplotlib
from grid2op.Plot.PlotPlotly import PlotPlotly
from grid2op.Plot.PlotPyGame import PlotPyGame
from grid2op.Plot.Plotting import Plotting
# from grid2op.Plot.EpisodeReplay import EpisodeReplay

import warnings


class PlotGraph(BasePlot):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        warnings.warn(
            "PlotGraph has been renamed to BasePlot"
            " -- The old name will be removed in future versions",
            category=PendingDeprecationWarning,
        )
