# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

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
