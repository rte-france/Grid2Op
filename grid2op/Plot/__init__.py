__all__ = [
    "BasePlot", 
    "PlotMatplotlib",
    "PlotPlotly",
    "PlotPyGame"
]

from grid2op.Plot.BasePlot import BasePlot
from grid2op.Plot.PlotMatplotlib import PlotMatplotlib
from grid2op.Plot.PlotPlotly import PlotPlotly
from grid2op.Plot.PlotPyGame import PlotPyGame

import warnings


class PlotGraph(BasePlot):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        warnings.warn("TopoAndRedispAction has been renamed to TopologyAndDispatchAction"
                      " -- The old name will be removed in future versions",
                      category=PendingDeprecationWarning)