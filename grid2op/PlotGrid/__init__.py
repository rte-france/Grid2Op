__all__ = [
    "NUKE_COLOR",
    "THERMAL_COLOR",
    "WIND_COLOR",
    "SOLAR_COLOR",
    "HYDRO_COLOR",
    "NUKE_ID",
    "THERMAL_ID",
    "WIND_ID",
    "SOLAR_ID",
    "HYDRO_ID",
    "TYPE_GEN",
    "COLOR_GEN",
    "BasePlot",
]

from grid2op.PlotGrid.config import *
from grid2op.PlotGrid.BasePlot import BasePlot

# Conditional exports for optional dependencies
try:
    from grid2op.PlotGrid.PlotMatplot import PlotMatplot

    __all__.append("PlotMatplot")
except ImportError:
    pass  # Silent fail because it is optional
try:
    from grid2op.PlotGrid.PlotPlotly import PlotPlotly

    __all__.append("PlotPlotly")
except ImportError:
    pass  # Silent fail because it is optional
