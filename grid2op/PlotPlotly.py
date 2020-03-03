"""
This module provide a simple class to represent an :class:`grid2op.Observation.Observation` as a plotly graph.

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
import cmath
import pdb

try:
    from .PlotGraph import BasePlot
except:
    from PlotGraph import BasePlot

try:
    import plotly.graph_objects as go
    import seaborn as sns
    can_plot = True
except Exception as e:
    can_plot = False
    pass

__all__ = ["PlotObs"]

# TODO add tests there


# Some utilities to plot substation, lines or get the color id for the colormap.
def draw_sub(pos, radius=50):
    """
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
        line_color="LightSeaGreen",
        fillcolor="lightgray"
    )
    return res


def get_col(rho):
    """
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
    if rho < 0.7:
        return 0
    if rho < 0.8:
        return 1
    if rho < 0.9:
        return 2
    if rho < 1.:
        return 3
    if rho < 1.1:
        return 5
    return 6


def draw_line(pos_sub_or, pos_sub_ex, rho, color_palette, status):
    """
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
        line=dict(
            color=color_palette[get_col(rho)] if status else "gray",
            dash=None if status else "dash"
        )
    )
    return res


class PlotObs(BasePlot):
    """
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
    def __init__(self, substation_layout, observation_space,
                 radius_sub=25.,
                 load_prod_dist=70.,
                 bus_radius=4.):
        """

        Parameters
        ----------
        substation_layout: ``list``
            List of tupe given the position of each of the substation of the powergrid.

        observation_space: :class:`grid2op.Observation.ObservationHelper`
            Observation space

        """
        BasePlot.__init__(self,
                          substation_layout=substation_layout,
                          observation_space=observation_space,
                          radius_sub=radius_sub,
                          load_prod_dist=load_prod_dist,
                          bus_radius=bus_radius)
        if not can_plot:
            raise RuntimeError("Impossible to plot as plotly cannot be imported. Please install \"plotly\" and "
                               "\"seaborn\" with \"pip install --update plotly seaborn\"")

        # define a color palette, whatever...
        sns.set()
        # pal = sns.dark_palette("palegreen")
        # pal = sns.color_palette("coolwarm", 7)
        # pal = sns.light_palette("red", 7)
        # self.cols = pal.as_hex()
        pal = sns.light_palette("darkred", 8)
        self.cols = pal.as_hex()[1:]

    def plot_observation(self, observation, fig=None):
        res = self.get_plot_observation(observation, fig=fig)
        return res

    def get_plot_observation(self, observation, fig=None):
        """
        Plot the given observation in the given figure.

        For now it represents information about load and generator active values.
        It also display dashed powerlines when they are disconnected and the color of each powerlines depends on
        its relative flow (its flow in amperes divided by its maximum capacity).

        If a substation counts only 1 bus, nothing specific is display. If it counts more, then buses are materialized
        by colored dot and lines will connect every object to its appropriate bus (with the proper color).

        Names of substation and objects are NOT displayed on this figure to lower the amount of information.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The observation to plot

        fig: :class:`plotly.graph_objects.Figure`
            The figure on which to plot the observation. Possibly ``None``, in this case a new figure is made.

        Returns
        -------
        res: :class:`plotly.graph_objects.Figure`
            The figure updated with the data from the new observation.
        """
        if fig is None:
            fig = go.Figure()

        # draw name of substation
        # fig.add_trace(go.Scatter(x=[el for el, _ in self._layout["substations"]],
        #                          y=[el for _, el in self._layout["substations"]],
        #                          text=["sub_{}".format(i) for i, _ in enumerate(self._layout["substations"])],
        #                          mode="text",
        #                          showlegend=False))

        # if not "line" in self._layout:
        #     # update the layout of the objects only once to ensure the same positionning is used
        #     # if more than 1 observation are displayed one after the other.
        #     self._compute_layout()

        # draw substation
        subs = self._draw_subs(observation=observation)
        # draw powerlines
        lines = self._draw_powerlines(observation, fig)
        # draw the loads
        loads = self._draw_loads(observation, fig)
        # draw the generators
        gens = self._draw_gens(observation, fig)
        # draw the topologies
        topos = self._draw_topos(observation, fig)
        # update the figure with all these information
        fig.update_layout(shapes=subs + lines + loads + gens + topos)

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
            plot_bgcolor="white"
        )
        return fig

    def _draw_sub(self, center):
        res = draw_sub(center, radius=self.radius_sub)
        return res

    def _draw_powerlines(self, observation, fig):
        """
        Draw the powerline, between two substations.

        Parameters
        ----------
        observation
        fig

        Returns
        -------

        """

        lines = []
        for line_id, (rho, status) in enumerate(zip(observation.rho, observation.line_status)):
            # compute the coordinates of the powerlines (coordinates of origin and extremity)
            pos_or, pos_ex, *_ = self._get_line_coord(line_id)

            # this depends on the grid
            # on this powergrid, thermal limit are not set at all. They are basically random.
            # so i multiply them by 300
            # rho *= 300
            lines.append(draw_line(pos_or, pos_ex,
                                   rho=rho,
                                   color_palette=self.cols,
                                   status=status))

            fig.add_trace(go.Scatter(x=[(pos_or[0] + pos_ex[0]) / 2],
                                     y=[(pos_or[1] + pos_ex[1]) / 2],
                                     text=["{:.1f}%".format(rho * 100)],
                                     mode="text",
                                     showlegend=False))
        return lines

    def _draw_loads(self, observation, fig):
        loads = []
        for c_id, por in enumerate(observation.load_p):
            pos_end_line, pos_load_sub, pos_load, how_center = self._get_load_coord(c_id)

            # add the MW load
            fig.add_trace(go.Scatter(x=[pos_load.real],
                                     y=[pos_load.imag],
                                     text=["- {:.0f} MW".format(por)],
                                     mode="text",
                                     showlegend=False))
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
                line=dict(
                )
            )
            loads.append(res)
        return loads

    def _draw_gens(self, observation, fig):
        gens = []
        for g_id, por in enumerate(observation.prod_p):
            pos_end_line, pos_gen_sub, pos_gen, how_center = self._get_gen_coord(g_id)

            # add the MW load
            fig.add_trace(go.Scatter(x=[pos_gen.real],
                                     y=[pos_gen.imag],
                                     text=["+ {:.0f} MW".format(por)],
                                     mode="text",
                                     showlegend=False))
            # add the line between the MW display and the substation
            # TODO later one, add something that looks like a load, a house for example
            res = go.layout.Shape(
                type="line",
                xref="x",
                yref="y",
                x0=pos_end_line.real,
                y0=pos_end_line.imag,
                x1=pos_gen_sub[0],
                y1=pos_gen_sub[1],
                line=dict(
                )
            )
            gens.append(res)
        return gens

    def _draw_topos(self, observation, fig):
        res_topo = []
        for sub_id, elements in enumerate(self.subs_elements):

            buses_z, bus_vect = self._get_topo_coord(sub_id, observation, elements)

            if not buses_z:
                # I don't plot details of substations with 1 bus for better quality
                continue

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
                res_topo.append(res)
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
                    res_topo.append(res)
        return res_topo
