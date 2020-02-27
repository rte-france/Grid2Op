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

import numpy as np
import cmath
import pdb

try:
    from .PlotGraph import BasePlot
except:
    from PlotGraph import BasePlot

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    can_plot = True
except Exception as e:
    can_plot = False
    pass

__all__ = ["GetLayout"]

# TODO add tests there
from grid2op.PlotGraph import BasePlot


class GetLayout(BasePlot):
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
            raise RuntimeError("Impossible to plot as matplotlib cannot be imported. Please install \"matplotlib\" "
                               " with \"pip install --update matplotlib\"")

    def plot_layout(self):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        col_line = "b"
        col_sub = "r"
        col_load = "k"
        col_gen = "g"

        legend_help = [Line2D([0], [0], color=col_line, lw=4),
                       Line2D([0], [0], color=col_sub, lw=4),
                       Line2D([0], [0], color=col_load, lw=4),
                       Line2D([0], [0], color=col_gen, lw=4)]

        # draw powerline
        for line_id in range(self.n_line):
            pos_or, pos_ex, *_ = self._get_line_coord(line_id)
            ax.plot([pos_or[0], pos_ex[0]], [pos_or[1], pos_ex[1]],
                    color=col_line)
            ax.text((pos_or[0] + pos_ex[0]) * 0.5,
                    (pos_or[1] + pos_ex[1]) * 0.5,
                    "{}\nid: {}".format(self.name_line[line_id], line_id),
                    color=col_line,
                    horizontalalignment='center',
                    verticalalignment='center')

        # draw substation
        for sub_id, center in enumerate(self._layout["substations"]):
            sub_circ = plt.Circle(center, self.radius_sub, color=col_sub, fill=False)
            ax.add_artist(sub_circ)
            ax.text(center[0],
                    center[1],
                    "{}\nid: {}".format(self.name_sub[sub_id], sub_id),
                    color=col_sub,
                    horizontalalignment='center',
                    verticalalignment='center')

        # draw loads
        for c_id in range(self.n_load):
            pos_end_line, pos_load_sub, pos_load, how_center = self._get_load_coord(c_id)
            ax.plot([pos_load_sub[0], pos_load.real],
                    [pos_load_sub[1], pos_load.imag],
                    color=col_load)
            ax.text(pos_load.real,
                    pos_load.imag,
                    "{}\nid: {}".format(self.name_load[c_id], c_id),
                    color=col_load,
                    horizontalalignment=how_center.split('|')[1],
                    verticalalignment="bottom" if how_center.split('|')[0] == "up" else "top")

        # draw gens
        for g_id in range(self.n_gen):
            pos_end_line, pos_gen_sub, pos_gen, how_center = self._get_gen_coord(g_id)
            ax.plot([pos_gen_sub[0], pos_gen.real],
                    [pos_gen_sub[1], pos_gen.imag],
                    color="g")
            ax.text(pos_gen.real,
                    pos_gen.imag,
                    "{}\nid: {}".format(self.name_gen[g_id], g_id),
                    color="g",
                    horizontalalignment=how_center.split('|')[1],
                    verticalalignment="bottom" if how_center.split('|')[0] == "up" else "top")
        ax.legend(legend_help, ["powerline", "substation", "load", "generator"])
        return fig

    def _draw_sub_layout(self, center, name):
        circle = plt.Circle((5, 5), 0.5, color='b', fill=False)
        return circle

    def _draw_powerlines(self, observation):

        for line_id, (rho, status, p_or) in enumerate(zip(observation.rho, observation.line_status, observation.p_or)):
            # the next 5 lines are always the same, for each observation, it makes sense to compute it once
            # and then reuse it

            pos_or, pos_ex, *_ = self._get_line_coord(line_id)

            if not status:
                # line is disconnected
                _draw_dashed_line(self.screen, pygame.Color(0, 0, 0), pos_or, pos_ex)
            else:
                # line is connected

                # step 0: compute thickness and color
                if rho < (self.rho_max / 1.5):
                    amount_green = 255 - int(255. * 1.5 * rho / self.rho_max)
                else:
                    amount_green = 0

                amount_red = int(255 - (50 + int(205. * rho / self.rho_max)))
                color = pygame.Color(amount_red, amount_green, 20)

                width = 1
                if rho > self.rho_max:
                    width = 4
                elif rho > 1.:
                    width = 3
                elif rho > 0.9:
                    width = 2
                width += 3

                # step 1: draw the powerline with right color and thickness
                pygame.draw.line(self.screen, color, pos_or, pos_ex, width)

                # step 2: draw arrows indicating current flows
                _draw_arrow(self.screen, color, pos_or, pos_ex,
                            p_or >= 0.,
                            num_arrows=width,
                            width=width)

    def _aligned_text(self, pos, text_graphic, pos_text):
        pos_x = pos_text.real
        pos_y = pos_text.imag
        width = text_graphic.get_width()
        height = text_graphic.get_height()

        if pos == "center|left":
            pos_y -= height // 2
        elif pos == "up|center":
            pos_x -= width // 2
            pos_y -= height
        elif pos == "center|right":
            pos_x -= width
            pos_y -= height // 2
        elif pos == "down|center":
            pos_x -= width // 2
        self.screen.blit(text_graphic, (pos_x, pos_y))

    def _draw_loads(self, observation):
        for c_id, por in enumerate(observation.load_p):
            pos_end_line, pos_load_sub, pos_load, how_center = self._get_load_coord(c_id)

            color = pygame.Color(0, 0, 0)
            width = 2
            pygame.draw.line(self.screen, color, pos_load_sub, (pos_end_line.real, pos_end_line.imag), width)
            text_label = "- {:.1f} MW".format(por)
            text_graphic = self.font.render(text_label, True, color)
            self._aligned_text(how_center, text_graphic, pos_load)

    def _draw_gens(self, observation):
        for g_id, por in enumerate(observation.prod_p):
            pos_end_line, pos_gen_sub, pos_gen, how_center = self._get_gen_coord(g_id)

            color = pygame.Color(0, 0, 0)
            width = 2
            pygame.draw.line(self.screen, color, pos_gen_sub, (pos_end_line.real, pos_end_line.imag), width)
            text_label = "+ {:.1f} MW".format(por)
            text_graphic = self.font.render(text_label, True, color)
            self._aligned_text(how_center, text_graphic, pos_gen)

    def _draw_topos(self, observation):
        for sub_id, elements in enumerate(self.subs_elements):
            buses_z, bus_vect = self._get_topo_coord(sub_id, observation, elements)

            if not buses_z:
                # I don't plot details of substations with 1 bus for better quality
                continue

            colors = [pygame.Color(255, 127, 14), pygame.Color(31, 119, 180)]

            # I plot the buses
            for bus_id, z_bus in enumerate(buses_z):
                pygame.draw.circle(self.screen,
                                   colors[bus_id],
                                   [int(z_bus.real), int(z_bus.imag)],
                                   int(self.bus_radius),
                                   0)

            # i connect every element to the proper bus with the proper color
            for el_nm, dict_el in elements.items():
                this_el_bus = bus_vect[dict_el["sub_pos"]] -1
                if this_el_bus >= 0:
                    pygame.draw.line(self.screen,
                                     colors[this_el_bus],
                                     [int(dict_el["z"].real), int(dict_el["z"].imag)],
                                     [int(buses_z[this_el_bus].real), int(buses_z[this_el_bus].imag)],
                                     2)


