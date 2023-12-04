# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import io
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path

from grid2op.PlotGrid.BasePlot import BasePlot
from grid2op.PlotGrid.PlotUtil import PlotUtil as pltu
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from grid2op.PlotGrid.config import *  # all colors


class GenDraw(patches.CirclePolygon):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Empty class to handle the legend
    """

    def __init__(self, *args, resolution=5, **kwargs):
        patches.CirclePolygon.__init__(self, *args, resolution=resolution, **kwargs)


class LoadDraw(patches.CirclePolygon):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Empty class to handle the legend
    """

    def __init__(self, *args, resolution=3, **kwargs):
        patches.CirclePolygon.__init__(self, *args, resolution=resolution, **kwargs)


class StorageDraw(patches.CirclePolygon):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Empty class to handle the legend
    """

    def __init__(self, *args, resolution=4, **kwargs):
        patches.CirclePolygon.__init__(self, *args, resolution=resolution, **kwargs)


# TODO refactor this class to make possible some calls like
# plotmatplot.plot_info(...).plot_gentype(...) is possible

# TODO add some transparency when coloring=... is used in plot_info
# TODO code the load part in the plot_info


class PlotMatplot(BasePlot):
    """
    This class uses the python library "matplotlib" to draw the powergrid.

    Attributes
    ----------

    width: ``int``
        Width of the figure in pixels
    height: ``int``
        Height of the figure in pixel
    dpi: ``int``
        Dots per inch, to convert pixels dimensions into inches
    _scale: ``float``
        Scale of the drawing in arbitrary units
    _sub_radius: ``int``
        Substation circle size
    _sub_face_color: ``str``
        Substation circle fill color
    _sub_edge_color: ``str``
        Substation circle edge color
    _sub_txt_color: ``str``
        Substation info text color
    _load_radius: ``int``
        Load circle size
    _load_name: ``bool``
        Show load names (default True)
    _load_face_color: ``str``
        Load circle fill color
    _load_edge_color: ``str``
        Load circle edge color
    _load_txt_color: ``str``
        Load info text color
    _load_line_color: ``str``
        Color of the line from load to substation
    _load_line_width: ``int``
        Width of the line from load to substation
    _gen_radius: ``int``
        Generators circle size
    _gen_name: ``bool``
        Show generators names (default True)
    _gen_face_color: ``str``
        Generators circle fill color
    _gen_edge_color: ``str``
        Generators circle edge color
    _gen_txt_color: ``str``
        Generators info txt color
    _gen_line_color: ``str``
        Color of the line form generator to substation
    _gen_line_width: ``str``
        Width of the line from generator to substation
    _line_color_scheme: ``list``
        List of color strings to color powerlines based on rho values
    _line_color_width: ``int``
        Width of the powerlines lines
    _line_bus_radius: ``int``
        Size of the bus display circle
    _line_bus_face_colors: ``list``
        List of 3 colors strings, each corresponding to the fill color of the bus circle
    _line_arrow_len: ``int``
        Length of the arrow on the powerlines
    _line_arrow_width: ``int``
       Width of the arrow on the powerlines

    Examples
    --------
    You can use it this way:

    .. code-block:: python

        import grid2op
        from grid2op.PlotGrid import PlotMatplot
        env = grid2op.make("l2rpn_case14_sandbox")
        plot_helper = PlotMatplot(env.observation_space)

        # and now plot an observation (for example)
        obs = env.reset()
        fig = plot_helper.plot_obs(obs)
        fig.show()

        # more information about it on the `getting_started/8_PlottingCapabilities.ipynb` notebook of grid2op

    """

    def __init__(
        self,
        observation_space,
        width=1280,
        height=720,
        grid_layout=None,
        dpi=96,
        scale=2000.0,
        bus_radius=6,
        sub_radius=15,
        load_radius=8,
        load_name=False,
        load_id=False,
        load_resolution=3,  # number of edges of the polygon representing the generator
        gen_radius=8,
        gen_name=False,
        gen_id=False,
        gen_resolution=5,  # number of edges of the polygon representing the generator
        storage_resolution=4,  # number of edges of the polygon representing the generator
        line_name=False,
        line_id=False,
    ):
        self.dpi = dpi
        super().__init__(observation_space, width, height, scale, grid_layout)

        self._sub_radius = sub_radius
        self._sub_face_color = "w"
        self._sub_edge_color = "blue"
        self._sub_txt_color = "black"
        self._display_sub_name = True

        self._load_radius = load_radius
        self._load_name = load_name
        self._load_id = load_id
        self._load_face_color = "w"
        self._load_edge_color = "orange"
        self._load_resolution = load_resolution
        self._load_patch = self._load_patch_default
        self._load_txt_color = "black"
        self._load_line_color = "black"
        self._load_line_width = 1
        self._display_load_name = True

        self._gen_radius_orig = gen_radius
        self._gen_radius = None  # init in self.restore_gen_palette()
        self._gen_resolution = gen_resolution
        self._gen_patch = self._gen_patch_default
        self._gen_name = gen_name
        self._gen_id = gen_id
        self._gen_face_color = "w"
        self._gen_edge_color_orig = "green"
        self._gen_edge_color = None
        self._gen_txt_color = "black"
        self._gen_line_color = "black"
        self._gen_line_width_orig = 1
        self._gen_line_width = None
        self._display_gen_value = True
        self._display_gen_name = True
        self.restore_gen_palette()

        self._storage_radius = load_radius
        self._storage_name = load_name
        self._storage_id = load_id  # bool : do i plot the id
        self._storage_face_color = "w"
        self._storage_edge_color = "purple"
        self._storage_resolution = storage_resolution
        self._storage_patch = self._storage_patch_default
        self._storage_txt_color = "black"
        self._storage_line_color = "black"
        self._storage_line_width = 1
        self._display_storage_name = True

        self._line_name = line_name
        self._line_id = line_id
        self._line_color_scheme_orig = ["blue", "orange", "red"]
        self._line_color_scheme = None
        self.restore_line_palette()

        self._line_color_width = 1
        self._line_bus_radius = bus_radius
        self._line_bus_face_colors = ["black", "red", "lime"]
        self._line_arrow_len = 10
        self._line_arrow_width = 10.0

        self.xlim = [0, 0]
        self.xpad = 5
        self.ylim = [0, 0]
        self.ypad = 5

        # for easize manipulation
        self.legend = None
        self.figure = None

    def _gen_patch_default(self, xy, radius, edgecolor, facecolor):
        """default patch used to draw generator"""
        # TODO maybe make a better version of this
        patch = GenDraw(
            xy,
            radius=radius,
            edgecolor=edgecolor,
            facecolor=facecolor,
            resolution=self._gen_resolution,
            linewidth=self._gen_line_width,
        )
        return patch

    def _load_patch_default(self, xy, radius, edgecolor, facecolor):
        """default patch used to draw generator"""
        # TODO maybe make a better version of this
        patch = LoadDraw(
            xy,
            radius=radius,
            edgecolor=edgecolor,
            facecolor=facecolor,
            resolution=self._load_resolution,
        )
        return patch

    def _storage_patch_default(self, xy, radius, edgecolor, facecolor):
        """default patch used to draw generator"""
        # TODO maybe make a better version of this
        patch = StorageDraw(
            xy,
            radius=radius,
            edgecolor=edgecolor,
            facecolor=facecolor,
            resolution=self._storage_resolution,
        )
        return patch

    def _v_textpos_from_dir(self, dirx, diry):
        if diry > 0:
            return "bottom"
        else:
            return "top"

    def _h_textpos_from_dir(self, dirx, diry):
        if dirx == 0:
            return "center"
        elif dirx > 0:
            return "left"
        else:
            return "right"

    def create_figure(self):
        # lazy loading of graphics library (reduce loading time)
        # and mainly because matplolib has weird impact on argparse
        import matplotlib.pyplot as plt

        w_inch = self.width / self.dpi
        h_inch = self.height / self.dpi
        f = plt.figure(figsize=(w_inch, h_inch), dpi=self.dpi)
        self.ax = f.subplots()
        f.canvas.draw()
        return f

    def clear_figure(self, figure):
        self.xlim = [0, 0]
        self.ylim = [0, 0]
        figure.clear()
        self.ax = figure.subplots()

    def convert_figure_to_numpy_HWC(self, figure):
        w, h = figure.get_size_inches() * figure.dpi
        w = int(w)
        h = int(h)
        buf = io.BytesIO()
        figure.canvas.print_raw(buf)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img_arr = np.reshape(img_arr, (h, w, 4))
        return img_arr

    def _draw_substation_txt(self, pos_x, pos_y, text):
        self.ax.text(
            pos_x,
            pos_y,
            text,
            color=self._sub_txt_color,
            horizontalalignment="center",
            verticalalignment="center",
        )

    def _draw_substation_circle(self, pos_x, pos_y):
        patch = patches.Circle(
            (pos_x, pos_y),
            radius=self._sub_radius,
            facecolor=self._sub_face_color,
            edgecolor=self._sub_edge_color,
        )
        self.ax.add_patch(patch)

    def draw_substation(self, figure, observation, sub_id, sub_name, pos_x, pos_y):
        self.xlim[0] = min(self.xlim[0], pos_x - self._sub_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._sub_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._sub_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._sub_radius)

        self._draw_substation_circle(pos_x, pos_y)
        if self._display_sub_name:
            self._draw_substation_txt(pos_x, pos_y, str(sub_id))

    def _draw_load_txt(self, pos_x, pos_y, sub_x, sub_y, text):
        dir_x, dir_y = pltu.vec_from_points(sub_x, sub_y, pos_x, pos_y)
        off_x, off_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        txt_x = pos_x + off_x * self._gen_radius
        txt_y = pos_y + off_y * self._gen_radius
        ha = self._h_textpos_from_dir(dir_x, dir_y)
        va = self._v_textpos_from_dir(dir_x, dir_y)
        self.ax.text(
            txt_x,
            txt_y,
            text,
            color=self._load_txt_color,
            horizontalalignment=ha,
            fontsize="small",
            verticalalignment=va,
        )

    def _draw_load_name(self, pos_x, pos_y, txt):
        self.ax.text(
            pos_x,
            pos_y,
            txt,
            color=self._load_txt_color,
            va="center",
            ha="center",
            fontsize="x-small",
        )

    def _draw_load_circle(self, pos_x, pos_y):
        patch = self._load_patch(
            (pos_x, pos_y),
            radius=self._load_radius,
            facecolor=self._load_face_color,
            edgecolor=self._load_edge_color,
        )
        self.ax.add_patch(patch)

    def _draw_load_line(self, pos_x, pos_y, sub_x, sub_y):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_x, pos_y), (sub_x, sub_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, color=self._load_line_color, lw=self._load_line_width
        )
        self.ax.add_patch(patch)

    def _draw_load_bus(self, pos_x, pos_y, norm_dir_x, norm_dir_y, bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle(
            (center_x, center_y), radius=self._line_bus_radius, facecolor=face_color
        )
        self.ax.add_patch(patch)

    def draw_load(
        self,
        figure,
        observation,
        load_id,
        load_name,
        load_bus,
        load_value,
        load_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        self.xlim[0] = min(self.xlim[0], pos_x - self._load_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._load_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._load_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._load_radius)
        self._draw_load_line(pos_x, pos_y, sub_x, sub_y)
        self._draw_load_circle(pos_x, pos_y)
        load_txt = ""
        if self._load_name:
            load_txt += '"{}":\n'.format(load_name)
        if self._load_id:
            load_txt += "id: {}\n".format(load_id)
        if load_value is not None:
            load_txt += pltu.format_value_unit(load_value, load_unit)
        if load_txt:
            self._draw_load_txt(pos_x, pos_y, sub_x, sub_y, load_txt)
        if self._display_load_name:
            self._draw_load_name(pos_x, pos_y, str(load_id))
        load_dir_x, load_dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        self._draw_load_bus(sub_x, sub_y, load_dir_x, load_dir_y, load_bus)

    def update_load(
        self,
        figure,
        observation,
        load_id,
        load_name,
        load_bus,
        load_value,
        load_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        pass

    def draw_storage(
        self,
        figure,
        observation,
        load_id,
        load_name,
        load_bus,
        load_value,
        load_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        self.xlim[0] = min(self.xlim[0], pos_x - self._load_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._load_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._load_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._load_radius)
        self._draw_storage_line(
            pos_x, pos_y, sub_x, sub_y
        )  # line from the storage to the substation
        self._draw_storage_circle(pos_x, pos_y)  # storage element

        load_txt = ""
        if self._storage_name:
            load_txt += '"{}":\n'.format(load_name)
        if self._storage_id:
            load_txt += "id: {}\n".format(load_id)
        if load_value is not None:
            load_txt += pltu.format_value_unit(load_value, load_unit)
        if load_txt:
            self._draw_load_txt(pos_x, pos_y, sub_x, sub_y, load_txt)
        if self._display_load_name:
            self._draw_load_name(pos_x, pos_y, str(load_id))
        load_dir_x, load_dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        self._draw_storage_bus(sub_x, sub_y, load_dir_x, load_dir_y, load_bus)

    def _draw_storage_circle(self, pos_x, pos_y):
        patch = self._storage_patch(
            (pos_x, pos_y),
            radius=self._storage_radius,
            facecolor=self._storage_face_color,
            edgecolor=self._storage_edge_color,
        )
        self.ax.add_patch(patch)

    def _draw_storage_line(self, pos_x, pos_y, sub_x, sub_y):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_x, pos_y), (sub_x, sub_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, color=self._storage_line_color, lw=self._storage_line_width
        )
        self.ax.add_patch(patch)

    def _draw_storage_bus(self, pos_x, pos_y, norm_dir_x, norm_dir_y, bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle(
            (center_x, center_y), radius=self._line_bus_radius, facecolor=face_color
        )
        self.ax.add_patch(patch)

    def update_storage(
        self,
        figure,
        observation,
        storage_name,
        storage_id,
        storage_bus,
        storage_value,
        storage_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        pass

    def _draw_gen_txt(self, pos_x, pos_y, sub_x, sub_y, text):
        dir_x, dir_y = pltu.vec_from_points(sub_x, sub_y, pos_x, pos_y)
        off_x, off_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        txt_x = pos_x + off_x * self._gen_radius
        txt_y = pos_y + off_y * self._gen_radius
        ha = self._h_textpos_from_dir(dir_x, dir_y)
        va = self._v_textpos_from_dir(dir_x, dir_y)
        self.ax.text(
            txt_x,
            txt_y,
            text,
            color=self._gen_txt_color,
            wrap=True,
            fontsize="small",
            horizontalalignment=ha,
            verticalalignment=va,
        )

    def _draw_gen_circle(self, pos_x, pos_y, gen_edgecolor):
        patch = self._gen_patch(
            (pos_x, pos_y),
            radius=self._gen_radius,
            edgecolor=gen_edgecolor,
            facecolor=self._gen_face_color,
        )
        self.ax.add_patch(patch)

    def _draw_gen_line(self, pos_x, pos_y, sub_x, sub_y):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_x, pos_y), (sub_x, sub_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, color=self._gen_line_color, lw=self._load_line_width
        )
        self.ax.add_patch(patch)

    def _draw_gen_name(self, pos_x, pos_y, txt):
        self.ax.text(
            pos_x,
            pos_y,
            txt,
            color=self._gen_txt_color,
            va="center",
            ha="center",
            fontsize="x-small",
        )

    def _draw_gen_bus(self, pos_x, pos_y, norm_dir_x, norm_dir_y, bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle(
            (center_x, center_y), radius=self._line_bus_radius, facecolor=face_color
        )
        self.ax.add_patch(patch)

    def draw_gen(
        self,
        figure,
        observation,
        gen_id,
        gen_name,
        gen_bus,
        gen_value,
        gen_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        self.xlim[0] = min(self.xlim[0], pos_x - self._gen_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._gen_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._gen_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._gen_radius)
        hide = False
        if isinstance(self._gen_edge_color, str):
            # case where the color of the generator is a string (same color for all generators)
            gen_color = self._gen_edge_color
        else:
            my_val = observation.prod_p[gen_id]
            n_colors = len(self._gen_edge_color) - 1
            if np.isfinite(my_val):
                color_idx = max(0, min(n_colors, int(my_val * n_colors)))
            else:
                color_idx = 0
                hide = True
            gen_color = self._gen_edge_color[color_idx]

        if not hide:
            self._draw_gen_line(pos_x, pos_y, sub_x, sub_y)
            self._draw_gen_circle(pos_x, pos_y, gen_color)
            gen_txt = ""
            if self._gen_name:
                gen_txt += '"{}":\n'.format(gen_name)
            if self._gen_id:
                gen_txt += "id: {}\n".format(gen_id)
            if gen_value is not None and self._display_gen_value:
                gen_txt += pltu.format_value_unit(gen_value, gen_unit)
            if gen_txt:
                self._draw_gen_txt(pos_x, pos_y, sub_x, sub_y, gen_txt)
            if self._display_gen_name:
                self._draw_gen_name(pos_x, pos_y, str(gen_id))
            gen_dir_x, gen_dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
            self._draw_gen_bus(sub_x, sub_y, gen_dir_x, gen_dir_y, gen_bus)

    def update_gen(
        self,
        figure,
        observation,
        gen_id,
        gen_name,
        gen_bus,
        gen_value,
        gen_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        pass

    def _draw_powerline_txt(self, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, text):
        pos_x, pos_y = pltu.middle_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        off_x, off_y = pltu.orth_norm_from_points(
            pos_or_x, pos_or_y, pos_ex_x, pos_ex_y
        )
        txt_x = pos_x + off_x * (self._load_radius / 2)
        txt_y = pos_y + off_y * (self._load_radius / 2)
        ha = self._h_textpos_from_dir(off_x, off_y)
        va = self._v_textpos_from_dir(off_x, off_y)
        self.ax.text(
            txt_x,
            txt_y,
            text,
            color=self._gen_txt_color,
            fontsize="small",
            horizontalalignment=ha,
            verticalalignment=va,
        )

    def _draw_powerline_line(
        self, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, color, line_style
    ):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_or_x, pos_or_y), (pos_ex_x, pos_ex_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, color=color, lw=self._line_color_width, ls=line_style
        )
        self.ax.add_patch(patch)

    def _draw_powerline_bus(self, pos_x, pos_y, norm_dir_x, norm_dir_y, bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle(
            (center_x, center_y), radius=self._line_bus_radius, facecolor=face_color
        )
        self.ax.add_patch(patch)

    def _draw_powerline_arrow(
        self, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, color, watt_value
    ):
        sign = 1.0 if watt_value > 0.0 else -1.0
        off = 1.0 if watt_value > 0.0 else 2.0
        dx, dy = pltu.norm_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        lx = dx * self._line_arrow_len
        ly = dy * self._line_arrow_len
        arr_x = pos_or_x + dx * self._sub_radius + off * lx
        arr_y = pos_or_y + dy * self._sub_radius + off * ly
        patch = patches.FancyArrow(
            arr_x,
            arr_y,
            sign * lx,
            sign * ly,
            length_includes_head=True,
            head_length=self._line_arrow_len,
            head_width=self._line_arrow_width,
            edgecolor=color,
            facecolor=color,
        )
        self.ax.add_patch(patch)

    def assign_line_palette(
        self, palette_name="YlOrRd", nb_color=10, line_color_scheme=None
    ):
        """
        Assign a new color palette when you want to plot information on the powerline.

        Parameters
        ----------
        palette_name: ``str``
            Name of the Maplotlib.plyplot palette to use (name forwarded to `plt.get_cmap(palette_name)`)
        nb_color: ``int``
            Number of color to use

        Examples
        -------
        .. code-block:: python

            # color a grid based on the value of the thermal limit
            plot_helper.assign_line_palette(nb_color=100)

            # plot this grid
            _ = plot_helper.plot_info(line_values=env.get_thermal_limit(), line_unit="A", coloring="line")

            # restore the default coloring (["blue", "orange", "red"])
            plot_helper.restore_line_palette()

        Notes
        -----
        Some palette are available there `colormaps <https://matplotlib.org/tutorials/colors/colormaps.html>`_

        """
        if line_color_scheme is None:
            palette = plt.get_cmap(palette_name)
            cols = []
            for i in range(1, nb_color + 1):
                cols.append(palette(i / nb_color))
            self._line_color_scheme = cols
        else:
            self._line_color_scheme = copy.deepcopy(line_color_scheme)

    def restore_line_palette(self):
        self._line_color_scheme = self._line_color_scheme_orig

    def assign_gen_palette(
        self,
        palette_name="YlOrRd",
        nb_color=10,
        increase_gen_size=None,
        gen_line_width=None,
    ):
        """
        Assign a new color palette when you want to plot information on the generator.

        Parameters
        ----------
        palette_name: ``str``
            Name of the Maplotlib.plyplot palette to use (name forwarded to `plt.get_cmap(palette_name)`)
        nb_color: ``int``
            Number of color to use

        increase_gen_size: ``float``
            Whether or not to increase the generator sizes (``None`` to disable this feature, 1 has no effect)

        gen_line_width: ``float``
            Increase the width of the generator (if not ``None``)

        Examples
        -------
        .. code-block:: python

            # color a grid based on the value of the thermal limit
            plot_helper.assign_gen_palette(nb_color=100)

            # plot this grid
            _ = plot_helper.plot_info(gen_values=env.gen_pmax, coloring="gen")

            # restore the default coloring (all green)
            plot_helper.restore_gen_palette()

        Notes
        -----
        Some palette are available there `colormaps <https://matplotlib.org/tutorials/colors/colormaps.html>`_

        """
        if palette_name is not None and nb_color > 0:
            # the user changed the palette
            palette = plt.get_cmap(palette_name)
            cols = []
            for i in range(1, nb_color + 1):
                cols.append(palette(i / nb_color))
            self._gen_edge_color = cols
        if increase_gen_size is not None:
            # the user changed the generator sizes
            self._gen_radius = float(increase_gen_size) * self._gen_radius_orig
        if gen_line_width is not None:
            # the user changed the generator line width
            self._gen_line_width = float(gen_line_width)

    def restore_gen_palette(self):
        """restore every properties of the default generator layout"""
        self._gen_edge_color = self._gen_edge_color_orig
        self._gen_radius = self._gen_radius_orig
        self._gen_line_width = self._gen_line_width_orig

    def draw_powerline(
        self,
        figure,
        observation,
        line_id,
        line_name,
        connected,
        line_value,
        line_unit,
        or_bus,
        pos_or_x,
        pos_or_y,
        ex_bus,
        pos_ex_x,
        pos_ex_y,
    ):
        rho = observation.rho[line_id]
        n_colors = len(self._line_color_scheme) - 1
        hide = False
        if np.isfinite(rho):
            color_idx = max(0, min(n_colors, int(rho * n_colors)))
        else:
            color_idx = 0
            hide = True

        color = "black"
        if not hide:
            if connected and rho > 0.0:
                color = self._line_color_scheme[color_idx]
            line_style = "-" if connected else "--"
            self._draw_powerline_line(
                pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, color, line_style
            )
            # Deal with line text configurations
            txt = ""
            if self._line_name:
                txt += '"{}"\n'.format(line_name)
            if self._line_id:
                txt += "id: {}\n".format(str(line_id))
            if line_value is not None:
                txt += pltu.format_value_unit(line_value, line_unit)
            if txt:
                self._draw_powerline_txt(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, txt)

            or_dir_x, or_dir_y = pltu.norm_from_points(
                pos_or_x, pos_or_y, pos_ex_x, pos_ex_y
            )
            self._draw_powerline_bus(pos_or_x, pos_or_y, or_dir_x, or_dir_y, or_bus)
            ex_dir_x, ex_dir_y = pltu.norm_from_points(
                pos_ex_x, pos_ex_y, pos_or_x, pos_or_y
            )
            self._draw_powerline_bus(pos_ex_x, pos_ex_y, ex_dir_x, ex_dir_y, ex_bus)
            watt_value = observation.p_or[line_id]
            if rho > 0.0 and watt_value != 0.0:
                self._draw_powerline_arrow(
                    pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, color, watt_value
                )

    def update_powerline(
        self,
        figure,
        observation,
        line_id,
        line_name,
        connected,
        line_value,
        line_unit,
        or_bus,
        pos_or_x,
        pos_or_y,
        ex_bus,
        pos_ex_x,
        pos_ex_y,
    ):
        pass

    def _get_gen_legend(self):
        """super complex function to display the proper shape in the legend"""
        if isinstance(self._gen_edge_color, str):
            gen_legend_col = self._gen_edge_color
        else:
            gen_legend_col = self._gen_edge_color[int(len(self._gen_edge_color) / 2)]
        my_res = self._gen_resolution

        class GenObjectHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                xdescent, ydescent = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                pp_ = GenDraw(
                    xy=center,
                    radius=min(width, height),
                    facecolor="w",
                    edgecolor=gen_legend_col,
                    transform=handlebox.get_transform(),
                    resolution=my_res,
                )
                handlebox.add_artist(pp_)
                return pp_

        gen_legend = self._gen_patch(
            (0, 0),
            facecolor=self._gen_face_color,
            edgecolor=gen_legend_col,
            radius=self._gen_radius,
        )
        return gen_legend, GenObjectHandler()

    def _get_load_legend(self):
        """super complex function to display the proper shape in the legend"""
        if isinstance(self._load_edge_color, str):
            load_legend_col = self._load_edge_color
        else:
            load_legend_col = self._load_edge_color[int(len(self._load_edge_color) / 2)]
        my_res = self._load_resolution

        class LoadObjectHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                xdescent, ydescent = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                pp_ = LoadDraw(
                    xy=center,
                    radius=min(width, height),
                    facecolor="w",
                    edgecolor=load_legend_col,
                    transform=handlebox.get_transform(),
                    resolution=my_res,
                )
                handlebox.add_artist(pp_)
                return pp_

        load_legend = self._load_patch(
            (0, 0),
            facecolor=self._load_face_color,
            edgecolor=load_legend_col,
            radius=self._load_radius,
        )
        return load_legend, LoadObjectHandler()

    def _get_storage_legend(self):
        """super complex function to display the proper shape in the legend"""
        if isinstance(self._storage_edge_color, str):
            storage_legend_col = self._storage_edge_color
        else:
            storage_legend_col = self._storage_edge_color[
                int(len(self._storage_edge_color) / 2)
            ]
        my_res = self._storage_resolution

        class StorageObjectHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                xdescent, ydescent = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                pp_ = StorageDraw(
                    xy=center,
                    radius=min(width, height),
                    facecolor="w",
                    edgecolor=storage_legend_col,
                    transform=handlebox.get_transform(),
                    resolution=my_res,
                )
                handlebox.add_artist(pp_)
                return pp_

        storage_legend = self._storage_patch(
            (0, 0),
            facecolor=self._storage_face_color,
            edgecolor=storage_legend_col,
            radius=self._storage_radius,
        )
        return storage_legend, StorageObjectHandler()

    def draw_legend(self, figure, observation):
        title_str = observation.env_name
        if hasattr(observation, "month"):
            title_str = "{:02d}/{:02d} {:02d}:{:02d}".format(
                observation.day,
                observation.month,
                observation.hour_of_day,
                observation.minute_of_hour,
            )

        # generate the right legend for generator
        gen_legend, gen_handler = self._get_gen_legend()
        # generate the correct legend for load
        load_legend, load_handler = self._get_load_legend()
        # generate the correct legend for storage
        storage_legend, storage_handler = self._get_storage_legend()

        legend_help = [
            Line2D([0], [0], color="black", lw=1),
            Line2D([0], [0], color=self._sub_edge_color, lw=3),
            load_legend,
            gen_legend,
            storage_legend,
            Line2D([0], [0], marker="o", color=self._line_bus_face_colors[0]),
            Line2D([0], [0], marker="o", color=self._line_bus_face_colors[1]),
            Line2D([0], [0], marker="o", color=self._line_bus_face_colors[2]),
        ]
        self.legend = self.ax.legend(
            legend_help,
            [
                "powerline",
                "substation",
                "load",
                "generator",
                "storage",
                "no bus",
                "bus 1",
                "bus 2",
            ],
            title=title_str,
            handler_map={
                GenDraw: gen_handler,
                LoadDraw: load_handler,
                StorageDraw: storage_handler,
            },
        )
        # Hide axis
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        # Hide frame
        self.ax.set(frame_on=False)

        # save the figure
        self.figure = figure

    def plot_postprocess(self, figure, observation, update):
        if not update:
            xmin = self.xlim[0] - self.xpad
            xmax = self.xlim[1] + self.xpad
            self.ax.set_xlim(xmin, xmax)
            ymin = self.ylim[0] - self.ypad
            ymax = self.ylim[1] + self.ypad
            self.ax.set_ylim(ymin, ymax)
            figure.tight_layout()

    def _save_plot_charact(self):
        _gen_edge_color_orig = self._gen_edge_color
        _gen_radius_orig = self._gen_radius
        _gen_line_width_orig = self._gen_line_width
        _display_gen_value = self._display_gen_value
        _display_gen_name = self._display_gen_name
        _display_sub_name = self._display_sub_name
        _display_load_name = self._display_load_name

        return (
            _gen_edge_color_orig,
            _gen_radius_orig,
            _gen_line_width_orig,
            _display_gen_value,
            _display_gen_name,
            _display_sub_name,
            _display_load_name,
        )

    def _restore_plot_charact(self, data):
        (
            _gen_edge_color_orig,
            _gen_radius_orig,
            _gen_line_width_orig,
            _display_gen_value,
            _display_gen_name,
            _display_sub_name,
            _display_load_name,
        ) = data
        self._gen_edge_color = _gen_edge_color_orig
        self._gen_radius = _gen_radius_orig
        self._gen_line_width = _gen_line_width_orig
        self._display_gen_value = _display_gen_value
        self._display_gen_name = _display_gen_name
        self._display_sub_name = _display_sub_name
        self._display_load_name = _display_load_name

    def plot_gen_type(self, increase_gen_size=1.5, gen_line_width=3):
        # save the sate of the generators config
        data = self._save_plot_charact()

        # do the plot
        self._display_gen_value = False
        self._display_gen_name = False
        self._display_sub_name = False
        self._display_load_name = False
        self.assign_gen_palette(
            nb_color=0,
            increase_gen_size=increase_gen_size,
            gen_line_width=gen_line_width,
        )
        self._gen_edge_color = [COLOR_GEN[i] for i in range(len(TYPE_GEN))]
        gen_values = [TYPE_GEN[el] for el in self.observation_space.gen_type]
        self.figure = self.plot_info(gen_values=gen_values, coloring="gen")
        self.add_legend_gentype()

        # restore the state to its initial configuration
        self._restore_plot_charact(data)

        return self.figure

    def plot_current_dispatch(
        self,
        obs,
        do_plot_actual_dispatch=True,
        increase_gen_size=1.5,
        gen_line_width=3,
        palette_name="coolwarm",
    ):
        # save the sate of the generators config
        data = self._save_plot_charact()

        # do the plot
        self._display_sub_name = False
        self._display_load_name = False
        self.assign_gen_palette(
            nb_color=5,
            palette_name=palette_name,
            increase_gen_size=increase_gen_size,
            gen_line_width=gen_line_width,
        )
        if do_plot_actual_dispatch:
            gen_values = obs.actual_dispatch
        else:
            gen_values = obs.target_dispatch
        self.figure = self.plot_info(
            gen_values=gen_values, coloring="gen", gen_unit="MW"
        )

        # restore the state to its initial configuration
        self._restore_plot_charact(data)

        return self.figure

    def add_legend_gentype(self, loc="lower right"):
        """add the legend for each generator type"""
        keys = sorted(TYPE_GEN.keys())
        ax_ = self.figure.axes[0]
        legend_help = [
            Line2D([0], [0], color=COLOR_GEN[TYPE_GEN[k]], label=k) for k in keys
        ]
        _ = ax_.legend(legend_help, keys, title="generator types", loc=loc)
        ax_.add_artist(self.legend)
