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
    from .Exceptions import PlotError
except (ModuleNotFoundError, ImportError):
    from PlotGraph import BasePlot
    from Exceptions import PlotError

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
    This class aims at simplifying the representation of the grid using matplotlib graphical libraries.

    It can be used to inspect position of elements, or to project some static data on this plot. It can be usefull
    to have a look at the thermal limit or the maximum value produced by generators etc.

    """

    def __init__(self,
                 substation_layout,
                 observation_space,
                 radius_sub=25.,
                 load_prod_dist=70.,
                 bus_radius=4.,
                 alpha_obj=0.3):
        BasePlot.__init__(self,
                          substation_layout=substation_layout,
                          observation_space=observation_space,
                          radius_sub=radius_sub,
                          load_prod_dist=load_prod_dist,
                          bus_radius=bus_radius)
        if not can_plot:
            raise RuntimeError("Impossible to plot as matplotlib cannot be imported. Please install \"matplotlib\" "
                               " with \"pip install --update matplotlib\"")

        self.alpha_obj = alpha_obj

        self.col_line = "b"
        self.col_sub = "r"
        self.col_load = "k"
        self.col_gen = "g"

    def plot_layout(self, figsize=(15, 15)):
        """
        This function plot the layout of the grid, as well as the object. You will see the name of each elements and
        their id.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        legend_help = [Line2D([0], [0], color=self.col_line, lw=4),
                       Line2D([0], [0], color=self.col_sub, lw=4),
                       Line2D([0], [0], color=self.col_load, lw=4),
                       Line2D([0], [0], color=self.col_gen, lw=4)]

        # draw powerline
        self._draw_powerlines(ax)

        # draw substation
        self._draw_subs(ax)

        # draw loads
        self._draw_loads(ax)

        # draw gens
        self._draw_gens(ax)
        ax.legend(legend_help, ["powerline", "substation", "load", "generator"])
        return fig

    def plot_info(self, line_info=None, load_info=None, gen_info=None, sub_info=None,
                  colormap=None):

        """
        Plot some information on the powergrid. For now, only numeric data are supported.

        Parameters
        ----------
        line_info: ``list``
            information to be displayed in the powerlines, in place of their name and id (for example their
            thermal limit) [must have the same size as the number of powerlines]
        load_info: ``list``
            information to display in the generators, in place of their name and id
            [must have the same size as the number of loads]
        gen_info: ``list``
            information to display in the generators, in place of their name and id (for example their pmax)
            [must have the same size as the number of generators]
        sub_info: ``list``
            information to display in the substation, in place of their name and id (for example the number of
            different topologies possible at this substation) [must have the same size as the number of substations]

        colormap: ``str``
            If not None, one of "line", "load", "gen" or "sub". If None, default colors will be used for each
            elements (default color is the coloring of
            If not None, all elements will be black, and the selected element will be highlighted.

        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        if colormap is None:
            legend_help = [Line2D([0], [0], color=self.col_line, lw=4),
                           Line2D([0], [0], color=self.col_sub, lw=4),
                           Line2D([0], [0], color=self.col_load, lw=4),
                           Line2D([0], [0], color=self.col_gen, lw=4)]

        # draw powerline
        texts_line = None
        if line_info is not None:
            texts_line = ["{:.2f}".format(el) if el is not None else None for el in line_info]
            if len(texts_line) != self.n_line:
                raise PlotError("Impossible to display these information on the powerlines: there are {} elements"
                                "provided while {} powerlines on this grid".format(len(texts_line), self.n_line))
        self._draw_powerlines(ax, texts_line, colormap=colormap)

        # draw substation
        texts_sub = None
        if sub_info is not None:
            texts_sub = ["{:.2f}".format(el) if el is not None else None for el in sub_info]
            if len(texts_sub) != self.n_sub:
                raise PlotError("Impossible to display these information on the substations: there are {} elements"
                                "provided while {} substations on this grid".format(len(texts_sub), self.n_sub))
        self._draw_subs(ax, texts_sub, colormap=colormap)

        # draw loads
        texts_load = None
        if load_info is not None:
            texts_load = ["{:.2f}".format(el) if el is not None else None for el in load_info]
            if len(texts_load) != self.n_load:
                raise PlotError("Impossible to display these information on the loads: there are {} elements"
                                "provided while {} loads on this grid".format(len(texts_load), self.n_load))
        self._draw_loads(ax, texts_load, colormap=colormap)

        # draw gens
        texts_gen = None
        if gen_info is not None:
            texts_gen = ["{:.2f}".format(el) if el is not None else None for el in gen_info]
            if len(texts_gen) != self.n_gen:
                raise PlotError("Impossible to display these information on the generators: there are {} elements"
                                "provided while {} generators on this grid".format(len(texts_gen), self.n_gen))
        self._draw_gens(ax, texts_gen, colormap=colormap)

        if colormap is None:
            ax.legend(legend_help, ["powerline", "substation", "load", "generator"])
        return fig

    def _draw_powerlines(self, ax, texts=None, colormap=None):
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
            ax.plot([pos_or[0], pos_ex[0]], [pos_or[1], pos_ex[1]],
                    color=this_col, alpha=self.alpha_obj)
            if text is not None:
                ax.text((pos_or[0] + pos_ex[0]) * 0.5,
                        (pos_or[1] + pos_ex[1]) * 0.5,
                        text,
                        color=this_col,
                        horizontalalignment='center',
                        verticalalignment='center')

    def _draw_subs(self, ax, texts=None, colormap=None):
        colormap_ = lambda x: self.col_sub
        vals = [0. for _ in self._layout["substations"]]

        if texts is not None:
            vals = [float(text if text is not None else 0.) for text in texts]

        if colormap is not None:
            colormap_ = lambda x: "k"
            if colormap == "sub":
                colormap_ = plt.get_cmap("Reds")
                vals = self._get_vals(vals)

        if texts is not None:
            vals = [float(text if text is not None else 0.) for text in texts]

        for sub_id, center in enumerate(self._layout["substations"]):
            if texts is None:
                text = "{}\nid: {}".format(self.name_sub[sub_id], sub_id)
                this_col = colormap_("")
            else:
                text = texts[sub_id]
                this_col = colormap_(vals[sub_id])
            sub_circ = plt.Circle(center, self.radius_sub, color=this_col, fill=False, alpha=self.alpha_obj)
            ax.add_artist(sub_circ)
            if text is not None:
                ax.text(center[0],
                        center[1],
                        text,
                        color=this_col,
                        horizontalalignment='center',
                        verticalalignment='center')

    def _draw_loads(self, ax, texts=None, colormap=None):
        colormap_ = lambda x: self.col_load
        vals = [0. for _ in range(self.n_load)]
        if texts is not None:
            vals = [float(text if text is not None else 0.) for text in texts]

        if colormap is not None:
            colormap_ = lambda x: "k"
            if colormap == "load":
                colormap_ = plt.get_cmap("Reds")
                vals = self._get_vals(vals)

        for c_id in range(self.n_load):
            if texts is None:
                text = "{}\nid: {}".format(self.name_load[c_id], c_id)
                this_col = colormap_(vals[c_id])
            else:
                text = texts[c_id]
                this_col = colormap_(float(text if text is not None else 0.))

            pos_end_line, pos_load_sub, pos_load, how_center = self._get_load_coord(c_id)
            ax.plot([pos_load_sub[0], pos_load.real],
                    [pos_load_sub[1], pos_load.imag],
                    color=this_col, alpha=self.alpha_obj)
            if text is not None:
                verticalalignment = self._getverticalalignment(how_center)
                ax.text(pos_load.real,
                        pos_load.imag,
                        text,
                        color=this_col,
                        horizontalalignment=how_center.split('|')[1],
                        verticalalignment=verticalalignment)

    def _getverticalalignment(self, how_center):
        verticalalignment = "center"
        if how_center.split('|')[0] == "up":
            verticalalignment = "bottom"
        elif how_center.split('|')[0] == "down":
            verticalalignment = "top"
        return verticalalignment

    def _get_vals(self, vals):
        min_ = np.min(vals)
        max_ = np.max(vals)
        vals -= min_
        vals /= (max_ - min_ + 1e-5)
        # now vals is between 0 and 1, i push it toward 1 a bit to better see it
        vals += 0.5
        vals /= 1.5
        return vals

    def _draw_gens(self, ax, texts=None, colormap=None):
        colormap_ = lambda x: self.col_gen
        vals = [0. for _ in range(self.n_gen)]
        if texts is not None:
            vals = [float(text if text is not None else 0.) for text in texts]

        if colormap is not None:
            colormap_ = lambda x: "k"
            if colormap == "gen":
                colormap_ = plt.get_cmap("Reds")
                vals = self._get_vals(vals)

        for g_id in range(self.n_gen):
            if texts is None:
                text = "{}\nid: {}".format(self.name_gen[g_id], g_id)
                this_col = colormap_("")
            else:
                text = texts[g_id]
                this_col = colormap_(vals[g_id])
            pos_end_line, pos_gen_sub, pos_gen, how_center = self._get_gen_coord(g_id)
            ax.plot([pos_gen_sub[0], pos_gen.real],
                    [pos_gen_sub[1], pos_gen.imag],
                    color=this_col, alpha=self.alpha_obj)
            if text is not None:
                verticalalignment = self._getverticalalignment(how_center)
                ax.text(pos_gen.real,
                        pos_gen.imag,
                        text,
                        color=this_col,
                        horizontalalignment=how_center.split('|')[1],
                        verticalalignment=verticalalignment)