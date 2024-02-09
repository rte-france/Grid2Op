# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import imageio
import warnings

# lazy loading of plotting utilities, to save loading time
import plotly.graph_objects as go
import plotly.colors as pc


from grid2op.PlotGrid.BasePlot import BasePlot
from grid2op.PlotGrid.PlotUtil import PlotUtil as pltu


class PlotPlotly(BasePlot):
    """

    This class uses the python library "plotly" to draw the powergrid. Plotly has the ability to generate
    interactive graphs.

    Examples
    --------
    You can use it this way:

    .. code-block:: python

        import grid2op
        from grid2op.PlotGrid import PlotPlotly
        env = grid2op.make("l2rpn_case14_sandbox")
        plot_helper = PlotPlotly(env.observation_space)

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
        responsive=False,
        scale=2000.0,
        sub_radius=25,
        load_radius=12,
        gen_radius=12,
        show_gen_txt=False,
        show_load_txt=False,
        show_storage_txt=False,
    ):
        super().__init__(observation_space, width, height, scale, grid_layout)
        self.show_gen_txt = show_gen_txt
        self.show_load_txt = show_load_txt
        self.show_storage_txt = show_storage_txt
        self._responsive = responsive
        self._sub_radius = sub_radius
        self._sub_fill_color = "PaleTurquoise"
        self._sub_line_color = "black"
        self._sub_line_width = 1
        self._sub_prefix = "z"

        self._load_radius = load_radius
        self._load_fill_color = "DarkOrange"
        self._load_line_color = "black"
        self._load_line_width = 1
        self._load_prefix = "c"

        self._gen_radius = gen_radius
        self._gen_fill_color = "LightGreen"
        self._gen_line_color = "black"
        self._gen_line_width = 1
        self._gen_prefix = "b"

        self._storage_radius = 12
        self._storage_fill_color = "Purple"
        self._storage_line_color = "black"
        self._storage_line_width = 1
        self._storage_prefix = "d"

        self._line_prefix = "a"
        self.line_color_scheme = (
            pc.sequential.Blues_r[:4]
            + pc.sequential.Oranges[4:6]
            + pc.sequential.Reds[-3:-1]
        )
        self._line_bus_radius = 10
        self._line_bus_colors = ["black", "red", "lime"]
        self._bus_prefix = "_bus_"
        self._or_prefix = "_or_"
        self._ex_prefix = "_ex_"
        self._line_arrow_radius = 10
        self._line_arrow_len = 5
        self._arrow_prefix = "_->_"

    def _textpos_from_dir(self, dirx, diry):
        typos = "bottom" if diry < 0 else "top"
        txpos = "left" if dirx < 0 else "right"
        return "{} {}".format(typos, txpos)

    def _set_layout(self, f):
        if not self._responsive:
            f.update_layout(
                width=self.width,
                height=self.height,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, b=0, t=0, pad=0),
            )
        else:
            f.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, b=0, t=0, pad=0),
            )

    def create_figure(self):
        f = go.Figure()
        self._set_layout(f)
        return f

    def clear_figure(self, figure):
        figure.layout = {}
        figure.data = []
        f = figure
        self._set_layout(f)

    def convert_figure_to_numpy_HWC(self, figure):
        try:
            img_bytes = figure.to_image(
                format="png", width=self.width, height=self.height, scale=1
            )
            return imageio.imread(img_bytes, format="png")
        except:
            warnings.warn("Plotly need additional dependencies for offline rendering")
            return np.full((self.height, self.width, 3), 255, dtype=np.unit8)

    def _draw_substation_txt(self, name, pos_x, pos_y, text):
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            text=[text],
            mode="text",
            name=name,
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
        )

    def _draw_substation_circle(self, name, pos_x, pos_y):
        marker_dict = dict(
            size=self._sub_radius,
            color=self._sub_fill_color,
            showscale=False,
            line=dict(width=self._sub_line_width, color=self._sub_line_color),
        )
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            mode="markers",
            text=[name],
            name=self._sub_prefix + name,
            marker=marker_dict,
            showlegend=False,
        )

    def draw_substation(self, figure, observation, sub_id, sub_name, pos_x, pos_y):
        circle_trace = self._draw_substation_circle(sub_name, pos_x, pos_y)
        figure.add_trace(circle_trace)
        txt_trace = self._draw_substation_txt(sub_name, pos_x, pos_y, str(sub_id))
        figure.add_trace(txt_trace)

    def _draw_load_txt(self, name, pos_x, pos_y, text, textpos):
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            text=[text],
            mode="text",
            name=name,
            hoverinfo="skip",
            textposition=textpos,
            showlegend=False,
        )

    def _draw_load_circle(self, pos_x, pos_y, name, text):
        marker_dict = dict(
            size=self._load_radius,
            color=self._load_fill_color,
            showscale=False,
            line=dict(width=self._load_line_width, color=self._load_line_color),
        )
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            mode="markers",
            text=[text],
            name=self._load_prefix + name,
            marker=marker_dict,
            showlegend=False,
        )

    def _draw_load_line(self, pos_x, pos_y, sub_x, sub_y):
        style_line = dict(color="black", width=self._load_line_width)

        line_trace = go.Scatter(
            x=[pos_x, sub_x],
            y=[pos_y, sub_y],
            hoverinfo="skip",
            line=style_line,
            showlegend=False,
        )
        return line_trace

    def _draw_load_bus(self, pos_x, pos_y, dir_x, dir_y, bus, load_name):
        bus = bus if bus > 0 else 0
        marker_dict = dict(
            size=self._line_bus_radius,
            color=self._line_bus_colors[bus],
            showscale=False,
        )
        center_x = pos_x + dir_x * (self._sub_radius - self._line_bus_radius)
        center_y = pos_y + dir_y * (self._sub_radius - self._line_bus_radius)
        trace_name = self._load_prefix + self._bus_prefix + load_name
        return go.Scatter(
            x=[center_x],
            y=[center_y],
            marker=marker_dict,
            name=trace_name,
            hoverinfo="skip",
            showlegend=False,
        )

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
        dir_x, dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        nd_x, nd_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        load_text = ""
        if load_value is not None:
            txt_x = pos_x + nd_x * (self._load_radius / 2)
            txt_y = pos_y + nd_y * (self._load_radius / 2)
            text_pos = self._textpos_from_dir(dir_x, dir_y)
            load_text = load_name + "<br>"
            load_text += pltu.format_value_unit(load_value, load_unit)
            if self.show_load_txt:
                trace1 = self._draw_load_txt(
                    load_name, txt_x, txt_y, load_text, text_pos
                )
                figure.add_trace(trace1)

        trace2 = self._draw_load_line(pos_x, pos_y, sub_x, sub_y)
        figure.add_trace(trace2)
        trace3 = self._draw_load_circle(pos_x, pos_y, load_name, load_text)
        figure.add_trace(trace3)

        trace4 = self._draw_load_bus(sub_x, sub_y, dir_x, dir_y, load_bus, load_name)
        figure.add_trace(trace4)

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
        load_text = ""
        if load_value is not None:
            load_text = load_name + "<br>"
            load_text += pltu.format_value_unit(load_value, load_unit)
            if self.show_load_txt:
                figure.update_traces(text=load_text, selector=dict(name=load_name))
            circle_name = self._load_prefix + load_name
            figure.update_traces(text=load_text, selector=dict(name=circle_name))
        load_marker = dict(color=self._line_bus_colors[load_bus])
        load_select_name = self._load_prefix + self._bus_prefix + load_name
        figure.update_traces(marker=load_marker, selector=dict(name=load_select_name))

    def _draw_gen_txt(self, name, pos_x, pos_y, text, text_pos):
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            text=[text],
            name=name,
            mode="text",
            hoverinfo="skip",
            textposition=text_pos,
            showlegend=False,
        )

    def _draw_gen_circle(self, pos_x, pos_y, name, text):
        marker_dict = dict(
            size=self._gen_radius,
            color=self._gen_fill_color,
            showscale=False,
            line=dict(width=self._gen_line_width, color=self._gen_line_color),
        )
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            mode="markers",
            text=[text],
            name=self._gen_prefix + name,
            marker=marker_dict,
            showlegend=False,
        )

    def _draw_gen_line(self, pos_x, pos_y, sub_x, sub_y):
        style_line = dict(color="black", width=self._gen_line_width)

        line_trace = go.Scatter(
            x=[pos_x, sub_x],
            y=[pos_y, sub_y],
            hoverinfo="skip",
            line=style_line,
            showlegend=False,
        )
        return line_trace

    def _draw_gen_bus(self, pos_x, pos_y, dir_x, dir_y, bus, gen_name):
        bus = bus if bus > 0 else 0
        marker_dict = dict(
            size=self._line_bus_radius,
            color=self._line_bus_colors[bus],
            showscale=False,
        )
        center_x = pos_x + dir_x * (self._sub_radius - self._line_bus_radius)
        center_y = pos_y + dir_y * (self._sub_radius - self._line_bus_radius)
        trace_name = self._gen_prefix + self._bus_prefix + gen_name
        return go.Scatter(
            x=[center_x],
            y=[center_y],
            marker=marker_dict,
            name=trace_name,
            hoverinfo="skip",
            showlegend=False,
        )

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

        dir_x, dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        nd_x, nd_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        gen_text = ""
        if gen_value is not None:
            txt_x = pos_x + nd_x * (self._gen_radius / 2)
            txt_y = pos_y + nd_y * (self._gen_radius / 2)
            text_pos = self._textpos_from_dir(dir_x, dir_y)
            gen_text = gen_name + "<br>"
            gen_text += pltu.format_value_unit(gen_value, gen_unit)
            if self.show_gen_txt:
                trace1 = self._draw_gen_txt(gen_name, txt_x, txt_y, gen_text, text_pos)
                figure.add_trace(trace1)
        trace2 = self._draw_gen_line(pos_x, pos_y, sub_x, sub_y)
        figure.add_trace(trace2)
        trace3 = self._draw_gen_circle(pos_x, pos_y, gen_name, gen_text)
        figure.add_trace(trace3)
        trace4 = self._draw_gen_bus(sub_x, sub_y, dir_x, dir_y, gen_bus, gen_name)
        figure.add_trace(trace4)

    def update_gen(
        self,
        figure,
        observation,
        gen_name,
        gen_id,
        gen_bus,
        gen_value,
        gen_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        gen_text = ""
        if gen_value is not None:
            gen_text = gen_name + "<br>"
            gen_text += pltu.format_value_unit(gen_value, gen_unit)
            if self.show_gen_txt:
                figure.update_traces(text=gen_text, selector=dict(name=gen_name))
            circle_name = self._gen_prefix + gen_name
            figure.update_traces(text=gen_text, selector=dict(name=circle_name))
        gen_marker = dict(color=self._line_bus_colors[gen_bus])
        gen_select_name = self._gen_prefix + self._bus_prefix + gen_name
        figure.update_traces(marker=gen_marker, selector=dict(name=gen_select_name))

    def _draw_powerline_txt(self, name, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, text):
        mid_x = (pos_or_x + pos_ex_x) / 2
        mid_y = (pos_or_y + pos_ex_y) / 2
        dir_x = pos_ex_x - pos_or_x
        dir_y = pos_ex_y - pos_or_y
        orth_x = -dir_y
        orth_y = dir_x
        orth_norm = np.linalg.norm([orth_x, orth_y])
        txt_x = mid_x + (orth_x / orth_norm) * 2
        txt_y = mid_y + (orth_y / orth_norm) * 2
        text_pos = self._textpos_from_dir(orth_x, orth_y)

        txt_trace = go.Scatter(
            x=[txt_x],
            y=[txt_y],
            text=[text],
            name=name,
            mode="text",
            textposition=text_pos,
            showlegend=False,
        )
        return txt_trace

    def _draw_powerline_line(self, name, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, style):

        line_trace = go.Scatter(
            x=[pos_or_x, pos_ex_x],
            y=[pos_or_y, pos_ex_y],
            name=self._line_prefix + name,
            line=style,
            hoverinfo="skip",
            showlegend=False,
        )
        return line_trace

    def _draw_powerline_bus(
        self, pos_x, pos_y, dir_x, dir_y, bus, line_name, side_prefix
    ):
        marker_dict = dict(
            size=self._line_bus_radius,
            color=self._line_bus_colors[bus],
            showscale=False,
        )
        center_x = pos_x + dir_x * (self._sub_radius - self._line_bus_radius)
        center_y = pos_y + dir_y * (self._sub_radius - self._line_bus_radius)
        trace_name = self._line_prefix + self._bus_prefix + side_prefix + line_name
        return go.Scatter(
            x=[center_x],
            y=[center_y],
            marker=marker_dict,
            name=trace_name,
            hoverinfo="skip",
            showlegend=False,
        )

    def _plotly_tri_from_line_dir_and_sign(self, dx, dy, sign):
        # One dimension dirs
        if dx >= -0.25 and dx <= 0.25:  # Vertical
            if (dy < 0.0 and sign > 0.0) or (dy > 0.0 and sign < 0.0):
                return "triangle-down"
            else:
                return "triangle-up"

        if dy >= -0.25 and dy <= 0.25:  # Horizontal
            if (dx < 0.0 and sign > 0.0) or (dx > 0.0 and sign < 0.0):
                return "triangle-left"
            else:
                return "triangle-right"

        # Two dimensions dirs
        if dx >= 0.0 and dy >= 0.0 and sign >= 0.0:  # NE
            return "triangle-ne"
        if dx >= 0.0 and dy >= 0.0 and sign < 0.0:  # NE * -1 = SW
            return "triangle-sw"
        if dx >= 0.0 and dy < 0.0 and sign >= 0.0:  # SE
            return "triangle-se"
        if dx >= 0.0 and dy < 0.0 and sign < 0.0:  # SE *-1 = NW
            return "triangle-nw"
        if dx < 0.0 and dy >= 0.0 and sign >= 0.0:  # NW
            return "triangle-nw"
        if dx < 0.0 and dy >= 0.0 and sign < 0.0:  # NW * -1 = SE
            return "triangle-se"
        if dx < 0.0 and dy < 0.0 and sign >= 0.0:  # SW
            return "triangle-sw"
        if dx < 0.0 and dy < 0.0 and sign < 0.0:  # SW*-1 = NE
            return "triangle-ne"

        return "triangle-up-dot"  # Should not be reached

    def _draw_powerline_arrow(
        self, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, watts_value, line_name, line_color
    ):
        cx, cy = pltu.middle_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        dx, dy = pltu.norm_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        sym = self._plotly_tri_from_line_dir_and_sign(dx, dy, watts_value)
        marker_dict = dict(
            size=self._line_arrow_radius, color=line_color, showscale=False, symbol=sym
        )

        sub_offx = dx * self._sub_radius
        sub_offy = dy * self._sub_radius
        or_offx = dx * self._line_arrow_len
        or_offy = dy * self._line_arrow_len
        arrx_or = pos_or_x + sub_offx + or_offx
        arrx_ex = pos_or_x + sub_offx
        arry_or = pos_or_y + sub_offy + or_offy
        arry_ex = pos_or_y + sub_offy
        trace_name = self._line_prefix + self._arrow_prefix + line_name
        return go.Scatter(
            x=[arrx_or, arrx_ex],
            y=[arry_or, arry_ex],
            hoverinfo="skip",
            showlegend=False,
            marker=marker_dict,
            name=trace_name,
        )

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

        color_scheme = self.line_color_scheme
        capacity = observation.rho[line_id]
        capacity = np.clip(capacity, 0.0, 1.0)
        color = color_scheme[int(capacity * float(len(color_scheme) - 1))]
        if np.abs(capacity) <= 1e-7:
            color = "black"
        line_style = dict(dash=None if connected else "dash", color=color)
        line_text = ""
        if line_value is not None:
            line_text = pltu.format_value_unit(line_value, line_unit)
            trace1 = self._draw_powerline_txt(
                line_name, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, line_text
            )
            figure.add_trace(trace1)
        trace2 = self._draw_powerline_line(
            line_name, pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, line_style
        )
        figure.add_trace(trace2)
        dir_x, dir_y = pltu.norm_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        trace3 = self._draw_powerline_bus(
            pos_or_x, pos_or_y, dir_x, dir_y, or_bus, line_name, self._or_prefix
        )
        trace4 = self._draw_powerline_bus(
            pos_ex_x, pos_ex_y, -dir_x, -dir_y, ex_bus, line_name, self._ex_prefix
        )
        figure.add_trace(trace3)
        figure.add_trace(trace4)
        watt_sign = observation.p_or[line_id]
        trace5 = self._draw_powerline_arrow(
            pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, watt_sign, line_name, color
        )
        figure.add_trace(trace5)

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
        color_scheme = self.line_color_scheme
        capacity = min(observation.rho[line_id], 1.0)
        color_idx = int(capacity * (len(color_scheme) - 1))
        color = color_scheme[color_idx]
        if np.abs(capacity) <= 1e-7:
            color = "black"
        if line_value is not None:
            line_text = pltu.format_value_unit(line_value, line_unit)
            figure.update_traces(text=line_text, selector=dict(name=line_name))

        line_style = dict(dash=None if connected else "dash", color=color)
        figure.update_traces(
            line=line_style, selector=dict(name=self._line_prefix + line_name)
        )

        or_bus = or_bus if or_bus > 0 else 0
        ex_bus = ex_bus if ex_bus > 0 else 0
        or_marker = dict(color=self._line_bus_colors[or_bus])
        ex_marker = dict(color=self._line_bus_colors[ex_bus])
        or_select_name = (
            self._line_prefix + self._bus_prefix + self._or_prefix + line_name
        )
        ex_select_name = (
            self._line_prefix + self._bus_prefix + self._ex_prefix + line_name
        )
        figure.update_traces(marker=or_marker, selector=dict(name=or_select_name))
        figure.update_traces(marker=ex_marker, selector=dict(name=ex_select_name))
        arrow_select_name = self._line_prefix + self._arrow_prefix + line_name
        watt_value = observation.p_or[line_id]
        dx, dy = pltu.norm_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        arrow_sym = self._plotly_tri_from_line_dir_and_sign(dx, dy, watt_value)
        arrow_display = True if capacity > 0.0 else False
        arrow_marker = dict(color=color, symbol=arrow_sym)
        figure.update_traces(
            marker=arrow_marker,
            visible=arrow_display,
            selector=dict(name=arrow_select_name),
        )

    def draw_legend(self, figure, observation):
        figure.update_layout(showlegend=False)


    def _draw_storage_txt(self, name, pos_x, pos_y, text, text_pos):
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            text=[text],
            name=name,
            mode="text",
            hoverinfo="skip",
            textposition=text_pos,
            showlegend=False,
        )

    def _draw_storage_circle(self, pos_x, pos_y, name, text):
        marker_dict = dict(
            size=self._storage_radius,
            color=self._storage_fill_color,
            showscale=False,
            line=dict(width=self._storage_line_width, color=self._storage_line_color),
        )
        return go.Scatter(
            x=[pos_x],
            y=[pos_y],
            mode="markers",
            text=[text],
            name=self._storage_prefix + name,
            marker=marker_dict,
            showlegend=False,
        )

    def _draw_storage_line(self, pos_x, pos_y, sub_x, sub_y):
        style_line = dict(color="black", width=self._storage_line_width)

        line_trace = go.Scatter(
            x=[pos_x, sub_x],
            y=[pos_y, sub_y],
            hoverinfo="skip",
            line=style_line,
            showlegend=False,
        )
        return line_trace

    def _draw_storage_bus(self, pos_x, pos_y, dir_x, dir_y, bus, storage_name):
        bus = bus if bus > 0 else 0
        marker_dict = dict(
            size=self._line_bus_radius,
            color=self._line_bus_colors[bus],
            showscale=False,
        )
        center_x = pos_x + dir_x * (self._sub_radius - self._line_bus_radius)
        center_y = pos_y + dir_y * (self._sub_radius - self._line_bus_radius)
        trace_name = self._storage_prefix + self._bus_prefix + storage_name
        return go.Scatter(
            x=[center_x],
            y=[center_y],
            marker=marker_dict,
            name=trace_name,
            hoverinfo="skip",
            showlegend=False,
        )

    def draw_storage(
        self,
        figure,
        observation,
        storage_id,
        storage_name,
        storage_bus,
        storage_value,
        storage_unit,
        pos_x,
        pos_y,
        sub_x,
        sub_y,
    ):
        # TODO storage doc
        dir_x, dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        nd_x, nd_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        storage_text = ""
        if storage_value is not None:
            txt_x = pos_x + nd_x * (self._storage_radius / 2)
            txt_y = pos_y + nd_y * (self._storage_radius / 2)
            text_pos = self._textpos_from_dir(dir_x, dir_y)
            storage_text = storage_name + "<br>"
            storage_text += pltu.format_value_unit(storage_value, storage_unit)
            if self.show_storage_txt:
                trace1 = self._draw_storage_txt(storage_name, txt_x, txt_y, storage_text, text_pos)
                figure.add_trace(trace1)
        trace2 = self._draw_storage_line(pos_x, pos_y, sub_x, sub_y)
        figure.add_trace(trace2)
        trace3 = self._draw_storage_circle(pos_x, pos_y, storage_name, storage_text)
        figure.add_trace(trace3)
        trace4 = self._draw_storage_bus(sub_x, sub_y, dir_x, dir_y, storage_bus, storage_name)
        figure.add_trace(trace4)

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

        storage_text = ""
        if storage_value is not None:
            storage_text = storage_name + "<br>"
            storage_text += pltu.format_value_unit(storage_value, storage_unit)
            figure.update_traces(text=storage_text, selector=dict(name=storage_name))
            circle_name = self._storage_prefix + storage_name
            if self.show_storage_txt:
                figure.update_traces(text=storage_text, selector=dict(name=circle_name))
        storage_marker = dict(color=self._line_bus_colors[storage_bus])
        storage_select_name = self._storage_prefix + self._bus_prefix + storage_name
        figure.update_traces(marker=storage_marker, selector=dict(name=storage_select_name))
