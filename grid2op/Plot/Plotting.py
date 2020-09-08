# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions import PlotError
from grid2op.Plot.PlotPlotly import PlotPlotly
from grid2op.Plot.PlotMatplotlib import PlotMatplotlib
from grid2op.Plot.PlotPyGame import PlotPyGame

from grid2op.Exceptions.PlotExceptions import PyGameQuit


class Plotting:
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

    """
    allwed_display_mod = {"pygame": PlotPyGame,
                          "plotly": PlotPlotly,
                          "matplotlib": PlotMatplotlib}

    def __init__(self,
                 observation_space,
                 display_mod="plotly",
                 substation_layout=None,
                 radius_sub=20.,
                 load_prod_dist=70.,
                 bus_radius=6.):
        if display_mod not in self.allwed_display_mod:
            raise PlotError("Only avaible plot mod are \"{}\". You specified \"{}\" which is not supported."
                            "".format(self.allwed_display_mod, display_mod))

        cls_ = self.allwed_display_mod[display_mod]
        self.displ_backend = cls_(observation_space,
                                  substation_layout=substation_layout,
                                  radius_sub=radius_sub,
                                  load_prod_dist=load_prod_dist,
                                  bus_radius=bus_radius)
        self.display_mod = display_mod

    def _display_fig(self, fig, display):
        if display:
            if self.display_mod == "plotly":
                fig.show()
            elif self.display_mod == "matplotlib":
                fig, ax = fig
                fig.show()

    def plot_layout(self, fig=None, reward=None, done=None, timestamp=None, display=True):
        try:
            fig = self.displ_backend.plot_layout(fig=fig, reward=reward, done=done, timestamp=timestamp)
            self._display_fig(fig, display=display)
        except PyGameQuit:
            pass
        return fig

    def plot_info(self, fig=None, line_info=None, load_info=None, gen_info=None, sub_info=None,
                  colormap=None, display=True):
        try:
            fig = self.displ_backend.plot_info(fig=fig, line_info=line_info, load_info=load_info, gen_info=gen_info,
                                               sub_info=sub_info, colormap=colormap)
            self._display_fig(fig, display=display)
        except PyGameQuit:
            pass
        return fig

    def plot_obs(self,
                 observation,
                 fig=None,
                 reward=None,
                 done=None,
                 timestamp=None,
                 line_info="rho",
                 load_info="p",
                 gen_info="p",
                 colormap="line",
                 display=True):
        try:
            fig = self.displ_backend.plot_obs(observation, fig=fig,
                                              reward=reward, done=done, timestamp=timestamp,
                                              line_info=line_info,
                                              load_info=load_info,
                                              gen_info=gen_info,
                                              colormap=colormap)
            self._display_fig(fig, display=display)
        except PyGameQuit:
            pass

        return fig