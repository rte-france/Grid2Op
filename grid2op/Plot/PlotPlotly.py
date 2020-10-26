# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This module provide a simple class to represent an :class:`grid2op.BaseObservation.BaseObservation` as a plotly graph.

We are aware that the graph can be largely improved. This tool is an example on what can be done with the Grid2Op
framework.

We hope It requires an valid installation of plotly and seaborn. These dependencies can be installed with:

.. code-block:: bash

    pip3 install grid2op[plots]

To use this plotting utilities, for example in a jupyter notebook please refer to the
``getting_started/4_StudyYourAgent`` notebook for more details information. The basic usage of this function is:

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

import numpy as np

from grid2op.Plot.BasePlot import BasePlot
from grid2op.Exceptions import PlotError

try:
    import plotly.graph_objects as go
    import seaborn as sns
    can_plot = True
except Exception as e:
    can_plot = False
    pass

# TODO add tests there


# Some utilities to plot substation, lines or get the color id for the colormap.
def draw_sub(pos, radius=50, line_color="LightSeaGreen"):
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

    This function will draw the contour of a unique substation.

    Parameters
    ----------
    pos: ``tuple``
        It represents the position (x,y) of the center of the substation

    radius: ``float``
        Positive floating point representing the "radius" of the substation.

    Returns
    -------
    res: :class:`plotly.graph_objects.layout.Shape`
        A representation, as a plotly object of the substation

    """
    pos_x, pos_y = pos
    res = go.layout.Shape(
        type="circle",
        xref="x",
        yref="y",
        x0=pos_x - radius,
        y0=pos_y - radius,
        x1=pos_x + radius,
        y1=pos_y + radius,
        line_color=line_color,
        layer="below"
    )
    return res


def get_col(rho):
    """
    .. warning:: /!\\\\ This class is deprecated /!\\\\

    Get the index (in the color palette) of the current  capacity usage.

    Parameters
    ----------
    rho: ``float``
        The capacity usage of a given powerline.

    Returns
    -------
    res: ``int``
        The integer (between 0 and 6) of this line capacity usage in terms of color.

    """
    if rho < 0.3:
        return 0
    if rho < 0.5:
        return 1
    if rho < 0.75:
        return 2
    if rho < 0.9:
        return 3
    if rho < 0.95:
        return 5
    return 6


def draw_line(pos_sub_or, pos_sub_ex, rho, color_palette, status, line_color="gray"):
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

    Draw a powerline with the color depending on its line capacity usage.

    Parameters
    ----------
    pos_sub_or: ``tuple``
        Position (x,y) of the origin end of the powerline

    pos_sub_ex: ``tuple``
        Position (x,y) of the extremity end of the powerline

    rho: ``float``
        Line capacity usage

    color_palette: ``object``
        The color palette to use

    status: ``bool``
        Powerline status (connected / disconnected). Disconnected powerlines are dashed.

    Returns
    -------
    res: :class:`plotly.graph_objects.layout.Shape`
        A representation, as a plotly object of the powerline

    """
    x_0, y_0 = pos_sub_or
    x_1, y_1 = pos_sub_ex

    res = go.layout.Shape(
        type="line",
        xref="x",
        yref="y",
        x0=x_0,
        y0=y_0,
        x1=x_1,
        y1=y_1,
        layer="below",
        line=dict(
            color=line_color,
            dash=None if status else "dash"
        )
    )
    return res


class PlotPlotly(BasePlot):
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

        Prefer using the class `grid2op.PlotGrid.PlotPlotly`

    This class aims at simplifying the representation of an observation as a plotly object given a layout of a given
    powergrid substation.
    It "automatically" handles the positionning of the powerlines, loads and generators based on that.

    This class is just here as an inspection tool. The results can be of course improved, epsecially the label of the
    powerlines, or the ppositioning of the loads and generators.

    Attributes
    ----------
    _layout: ``dict``
        Initial layout of the powergrid.

    subs_elements: ``list``
        For each substation, it gives a representation of all the object connected to it. So, for each substation, it
        has a dictionnary with:

            - key: the name of the objects
            - value: a dictionnary representing this object containing:

                - "type" : its type, among "load", "gen" and "line"
                - "sub_pos" (``int``) and index representing which element of the substation represents this object
                - "pos" : its position as a tuple
                - "z": its position as a complex number

    cols: ``object``
        A color palette, this should not be changed for now.

    radius_sub: ``float``
        The radius of each substation. The bigger this number, the better the topology will be visible, but the more
        space taken on the overall plot

    load_prod_dist: ``float``
        The distance between a load and a generator from the center of the substation. This must be higher than
        :attr:`PlotObs.radius_sub`

    bus_radius: ``float``
        The radius of the bus. When multiple buses are present in a substation, they are materialized by a filled
        circle. This number represents the size of these circles.



    """
    def __init__(self,
                 observation_space,
                 substation_layout=None,
                 radius_sub=25.,
                 load_prod_dist=70.,
                 bus_radius=4.):
        """

        Parameters
        ----------
        substation_layout: ``list``
            List of tupe given the position of each of the substation of the powergrid.

        observation_space: :class:`grid2op.Observation.ObservationSpace`
            BaseObservation space

        """
        BasePlot.__init__(self,
                          substation_layout=substation_layout,
                          observation_space=observation_space,
                          radius_sub=radius_sub,
                          load_prod_dist=load_prod_dist,
                          bus_radius=bus_radius)
        if not can_plot:
            raise PlotError("Impossible to plot as plotly cannot be imported. Please install \"plotly\" and "
                            "\"seaborn\" with \"pip install --update plotly seaborn\"")

        # define a color palette, whatever...
        sns.set()
        pal = sns.light_palette("darkred", 8)
        self.cols = pal.as_hex()[1:]

        self.col_line = "royalblue"
        self.col_sub = "red"
        self.col_load = "black"
        self.col_gen = "darkgreen"
        self.default_color = "black"
        self.type_fig_allowed = go.Figure

    def init_fig(self, fig, reward, done, timestamp):
        if fig is None:
            fig = go.Figure()
        elif not isinstance(fig, self.type_fig_allowed):
            raise PlotError("PlotPlotly cannot plot on figure of type {}. The accepted type is {}. You provided an "
                            "invalid argument for \"fig\"".format(type(fig), self.type_fig_allowed))
        return fig

    def _post_process_obs(self, fig, reward, done, timestamp, subs, lines, loads, gens, topos):
        # update the figure with all these information
        traces = []
        subs_el = []
        lines_el = []
        loads_el = []
        gens_el = []
        topos_el = []
        for el, trace_ in subs:
            subs_el.append(el)
            traces.append(trace_)
        for el, trace_ in lines:
            lines_el.append(el)
            traces.append(trace_)
        for el, trace_ in loads:
            loads_el.append(el)
            traces.append(trace_)
        for el, trace_ in gens:
            gens_el.append(el)
            traces.append(trace_)
        for el, _ in topos:
            topos_el.append(el)
            topos_el.append(el)
            # traces.append(trace_)
        fig.update_layout(shapes=subs_el + lines_el + loads_el + gens_el + topos_el)

        for trace_ in traces:
            fig.add_trace(trace_)

        # update legend, background color, size of the plot etc.
        fig.update_xaxes(range=[np.min([el for el, _ in self._layout["substations"]]) - 1.5 * (self.radius_sub +
                                                                                               self.load_prod_dist),
                                np.max([el for el, _ in self._layout["substations"]]) + 1.5 * (self.radius_sub +
                                                                                               self.load_prod_dist)],
                         zeroline=False)
        fig.update_yaxes(range=[np.min([el for _, el in self._layout["substations"]]) - 1.5 * (self.radius_sub +
                                                                                               self.load_prod_dist),
                                np.max([el for _, el in self._layout["substations"]]) + 1.5 * (self.radius_sub +
                                                                                               self.load_prod_dist)])
        fig.update_layout(
            margin=dict(
                l=20,
                r=20,
                b=100
            ),
            height=600,
            width=800,
            plot_bgcolor="white",
            yaxis={'showgrid': False, "showline": False, "zeroline": False},
            xaxis={'showgrid': False, "showline": False, "zeroline": False}
        )
        return fig

    def _draw_subs_one_sub(self, fig, sub_id, center, this_col, txt_):
        trace = go.Scatter(x=[center[0]],
                           y=[center[1]],
                           text=[txt_],
                           mode="text",
                           showlegend=False,
                           textfont=dict(
                               color=this_col
                           ))
        res = draw_sub(center, radius=self.radius_sub, line_color=this_col)
        return res, trace

    def _draw_powerlines_one_powerline(self, fig, l_id, pos_or, pos_ex, status, value, txt_, or_to_ex, this_col):
        """
        .. warning:: /!\\\\ This class is deprecated /!\\\\

            Prefer using the class `grid2op.PlotGrid.PlotPlotly`

        Draw the powerline, between two substations.

        Parameters
        ----------
        observation
        fig

        Returns
        -------

        """
        tmp = draw_line(pos_or,
                        pos_ex,
                        rho=value,
                        color_palette=self.cols,
                        status=status,
                        line_color=this_col
                        )
        trace = go.Scatter(x=[(pos_or[0] + pos_ex[0]) / 2],
                           y=[(pos_or[1] + pos_ex[1]) / 2],
                           text=[txt_],
                           mode="text",
                           showlegend=False,
                           textfont=dict(
                               color=this_col
                           ))
        return tmp, trace

    def _draw_loads_one_load(self, fig, l_id, pos_load, txt_, pos_end_line, pos_load_sub, how_center, this_col):
        # add the MW load
        trace = go.Scatter(x=[pos_load.real],
                           y=[pos_load.imag],
                           text=[txt_],
                           mode="text",
                           showlegend=False,
                           textfont=dict(
                               color=this_col
                           ))
        # add the line between the MW display and the substation
        # TODO later one, add something that looks like a load, a house for example
        res = go.layout.Shape(
            type="line",
            xref="x",
            yref="y",
            x0=pos_end_line.real,
            y0=pos_end_line.imag,
            x1=pos_load_sub[0],
            y1=pos_load_sub[1],
            layer="below",
            line=dict(color=this_col
            )
        )
        return res, trace

    def _draw_gens_one_gen(self, fig, g_id, pos_gen, txt_, pos_end_line, pos_gen_sub, how_center, this_col):
        # add the MW load
        trace = go.Scatter(x=[pos_gen.real],
                           y=[pos_gen.imag],
                           text=[txt_],
                           mode="text",
                           showlegend=False,
                           textfont=dict(
                               color=this_col
                           ))
        # add the line between the MW display and the substation
        # TODO later one, add something that looks like a generator, and could depend on the type of it!
        res = go.layout.Shape(
            type="line",
            xref="x",
            yref="y",
            x0=pos_end_line.real,
            y0=pos_end_line.imag,
            x1=pos_gen_sub[0],
            y1=pos_gen_sub[1],
            layer="below",
            line=dict(color=this_col
            )
        )
        return res, trace

    def _draw_topos_one_sub(self, fig, sub_id, buses_z, elements, bus_vect):
        res_sub = []
        # I plot the buses
        for bus_id, z_bus in enumerate(buses_z):
            bus_color = '#ff7f0e' if bus_id == 0 else '#1f77b4'
            res = go.layout.Shape(
                type="circle",
                xref="x",
                yref="y",
                x0=z_bus.real - self.bus_radius,
                y0=z_bus.imag - self.bus_radius,
                x1=z_bus.real + self.bus_radius,
                y1=z_bus.imag + self.bus_radius,
                fillcolor=bus_color,
                line_color=bus_color,
            )
            res_sub.append((res, None))
        # i connect every element to the proper bus with the proper color
        for el_nm, dict_el in elements.items():
            this_el_bus = bus_vect[dict_el["sub_pos"]] -1
            if this_el_bus >= 0:
                res = go.layout.Shape(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=dict_el["z"].real,
                    y0=dict_el["z"].imag,
                    x1=buses_z[this_el_bus].real,
                    y1=buses_z[this_el_bus].imag,
                    line=dict(color='#ff7f0e' if this_el_bus == 0 else '#1f77b4'))
                res_sub.append((res, None))
        return res_sub

    def _get_default_cmap(self, normalized_value):
        return self.cols[get_col(normalized_value)]