__all__ = ["PlotObs"]

from grid2op.Plot.PlotPlotly import PlotPlotly
import warnings

warnings.warn("The grid2op.PlotPlotly module is deprecated and will be removed in a future version. Please "
              "update your code to use: \"from grid2op.Plot import PlotPlotly\" instead of "
              "\"from grid2op.PlotPlotly import PlotObs\".",
              category=PendingDeprecationWarning)


class PlotObs(PlotPlotly):
    def __init__(self, *args, **kwargs):
        PlotPlotly.__init__(self, *args, **kwargs)
        warnings.warn("grid2op.PlotPlotly.PlotObs class has been moved to  \"grid2op.Plot.PlotPlotly\" "
                      "for a better overall consistency of the grid2op package. "
                      "This module will be removed in future versions.",
                      category=PendingDeprecationWarning)
