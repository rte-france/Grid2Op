# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
TODO

.. code-block:: python

    # make the relevant import
    from grid2op.MakeEnv import make
    from grid2op.PlotPlotly import PlotObs

    # create a simple toy environment
    environment = make("case5_example")

    # set up the plot utility
    graph_layout =  [(0,0), (0,400), (200,400), (400, 400), (400, 0)]
    plot_helper = PlotObs(substation_layout=graph_layout,
                          observation_space=environment.observation_space)

    # perform a step from this environment:
    do_nothing = environment.action_space({})
    environment.step(act)

    # do the actual plot
    fig = plot_helper.get_plot_observation(environment.get_obs())
    fig.show()

"""
import warnings

from grid2op.Exceptions import PlotError
from grid2op.Plot.BasePlot import BasePlot

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    can_plot = True
except Exception as e:
    can_plot = False
    pass

# TODO add tests there


class PlotMatplotlib(BasePlot):
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

        Prefer using the class `grid2op.PlotGrid.PlotMatplot`

    This class aims at simplifying the representation of the grid using matplotlib graphical libraries.

    It can be used to inspect position of elements, or to project some static data on this plot. It can be usefull
    to have a look at the thermal limit or the maximum value produced by generators etc.

    """

    def __init__(self,
                 observation_space,
                 substation_layout=None,
                 radius_sub=25.,
                 load_prod_dist=70.,
                 bus_radius=4.,
                 alpha_obj=0.3,
                 figsize=(15, 15)):
        BasePlot.__init__(self,
                          substation_layout=substation_layout,
                          observation_space=observation_space,
                          radius_sub=radius_sub,
                          load_prod_dist=load_prod_dist,
                          bus_radius=bus_radius)

        warnings.warn("This whole class has been deprecated. Use `grid2op.PlotGrid.PlotMatplot` instead`",
                      category=DeprecationWarning)

        if not can_plot:
            raise RuntimeError("Impossible to plot as matplotlib cannot be imported. Please install \"matplotlib\" "
                               " with \"pip install --update matplotlib\"")

        self.alpha_obj = alpha_obj

        self.col_line = "b"
        self.col_sub = "r"
        self.col_load = "k"
        self.col_gen = "g"
        self.figsize = figsize
        self.default_color = "k"

        self.my_cmap = plt.get_cmap("Reds")
        self.accepted_figure_class = matplotlib.figure.Figure
        self.accepted_figure_class_tuple = (matplotlib.figure.Figure, matplotlib.axes.Axes)

    def init_fig(self, fig, reward, done, timestamp):
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        elif isinstance(fig, tuple):
            if len(fig) != 2:
                raise PlotError("PlotMatplotlib \"fig\" argument should be, if a tuple, a tuple containing a figure "
                                "and an axe, for example the results of `plt.subplots(1, 1)`. You provided "
                                "a tuple of length {}".format(len(fig)))
            fig, ax = fig
            if not isinstance(fig, self.accepted_figure_class):
                raise PlotError("PlotMatplotlib \"fig\" argument should be an object of type \"{}\" and not \"{}\"."
                                "".format(self.accepted_figure_class, type(fig)))
            if not isinstance(ax, self.accepted_figure_class_tuple[1]):
                raise PlotError("PlotMatplotlib \"fig\" argument should be an object of type \"{}\" and not \"{}\"."
                                "".format(self.accepted_figure_class, type(ax)))
        elif isinstance(fig, self.accepted_figure_class):
            ax = fig.gca()
        else:
            raise PlotError("PlotMatplotlib \"fig\" argument should be an object of type \"{}\" and not \"{}\"."
                                "".format(self.accepted_figure_class, type(fig)))
        return (fig, ax)

    def post_process_layout(self, fig, subs, lines, loads, gens, topos):
        legend_help = [Line2D([0], [0], color=self.col_line, lw=4),
                       Line2D([0], [0], color=self.col_sub, lw=4),
                       Line2D([0], [0], color=self.col_load, lw=4),
                       Line2D([0], [0], color=self.col_gen, lw=4)]
        fig, ax = fig
        ax.legend(legend_help, ["powerline", "substation", "load", "generator"])

    def _getverticalalignment(self, how_center):
        verticalalignment = "center"
        if how_center.split('|')[0] == "up":
            verticalalignment = "bottom"
        elif how_center.split('|')[0] == "down":
            verticalalignment = "top"
        return verticalalignment

    def _draw_loads_one_load(self, fig, l_id, pos_load, txt_, pos_end_line, pos_load_sub, how_center, this_col):
        fig, ax = fig
        ax.plot([pos_load_sub[0], pos_load.real],
                [pos_load_sub[1], pos_load.imag],
                color=this_col, alpha=self.alpha_obj)
        if txt_ is not None:
            verticalalignment = self._getverticalalignment(how_center)
            ax.text(pos_load.real,
                    pos_load.imag,
                    txt_,
                    color=this_col,
                    horizontalalignment=how_center.split('|')[1],
                    verticalalignment=verticalalignment)

    def _draw_gens_one_gen(self, fig, g_id, pos_gen, txt_, pos_end_line, pos_gen_sub, how_center, this_col):
        fig, ax = fig
        pos_end_line_, pos_gen_sub_, pos_gen_, how_center_ = self._get_gen_coord(g_id)
        ax.plot([pos_gen_sub_[0], pos_gen_.real],
                [pos_gen_sub_[1], pos_gen_.imag],
                color=this_col, alpha=self.alpha_obj)
        if txt_ is not None:
            verticalalignment = self._getverticalalignment(how_center_)
            ax.text(pos_gen_.real,
                    pos_gen_.imag,
                    txt_,
                    color=this_col,
                    horizontalalignment=how_center_.split('|')[1],
                    verticalalignment=verticalalignment)

    def _draw_powerlines_one_powerline(self, fig, l_id, pos_or, pos_ex, status, value, txt_, or_to_ex, this_col):
        fig, ax = fig
        ax.plot([pos_or[0], pos_ex[0]],
                [pos_or[1], pos_ex[1]],
                color=this_col,
                alpha=self.alpha_obj,
                linestyle="solid" if status else "dashed")

        if txt_ is not None:
            ax.text((pos_or[0] + pos_ex[0]) * 0.5,
                    (pos_or[1] + pos_ex[1]) * 0.5,
                     txt_,
                     color=this_col,
                     horizontalalignment='center',
                     verticalalignment='center')

    def _draw_subs_one_sub(self, fig, sub_id, center, this_col, text):
        fig, ax = fig
        sub_circ = plt.Circle(center, self.radius_sub, color=this_col, fill=False)  #, alpha=self.alpha_obj)
        ax.add_artist(sub_circ)
        if text is not None:
            ax.text(center[0],
                    center[1],
                    text,
                    color=this_col,
                    horizontalalignment='center',
                    verticalalignment='center')

    def _get_default_cmap(self, normalized_val):
        return self.my_cmap(normalized_val)

    def _draw_topos_one_sub(self, fig, sub_id, buses_z, elements, bus_vect):
        fig, ax = fig
        res_sub = []
        # I plot the buses
        for bus_id, z_bus in enumerate(buses_z):
            bus_color = '#ff7f0e' if bus_id == 0 else '#1f77b4'
            bus_circ = plt.Circle((z_bus.real, z_bus.imag), self.bus_radius, color=bus_color, fill=True)
            ax.add_artist(bus_circ)

        # i connect every element to the proper bus with the proper color
        for el_nm, dict_el in elements.items():
            this_el_bus = bus_vect[dict_el["sub_pos"]] -1
            if this_el_bus >= 0:
                color = '#ff7f0e' if this_el_bus == 0 else '#1f77b4'
                ax.plot([buses_z[this_el_bus].real, dict_el["z"].real],
                        [ buses_z[this_el_bus].imag, dict_el["z"].imag],
                        color=color, alpha=self.alpha_obj)
        return []

    def _draw_powerlines____________(self, ax, texts=None, colormap=None):
        colormap_ = lambda x: self.col_line
        vals = [0. for _ in range(self.n_line)]
        if texts is not None:
            vals = [float(text if text is not None else 0.) for text in texts]

        if colormap is not None:
            colormap_ = lambda x: "k"
            if colormap == "line":
                colormap_ = plt.get_cmap("Reds")
                vals = self._get_vals(vals)

        for line_id in range(self.n_line):
            if texts is None:
                text = "{}\nid: {}".format(self.name_line[line_id], line_id)
                this_col = colormap_("")
            else:
                text = texts[line_id]
                this_col = colormap_(vals[line_id])


            pos_or, pos_ex, *_ = self._get_line_coord(line_id)

