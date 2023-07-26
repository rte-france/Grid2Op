# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This module is the base module for all graphical representation of the :class:`grid2op.BaseObservation.BaseObservation`.

It allows, from the layout of the graph of the powergrid (*eg* the coordinates of each substation) the position of
each objects (powerline ends, loads and generators) and the position of the buses in case of "node splitting" (when
a substations is split into independent electrical buses).

"""
import cmath
import math
import numpy as np
import warnings
import copy

from grid2op.Space import GridObjects
from grid2op.Exceptions import PlotError


class BasePlot(GridObjects):
    """
    INTERNAL

    .. warning:: /!\\\\ This module is deprecated /!\\\\

        Prefer using the module `grid2op.PlotGrid

    Utility class that allows to compute the position of the objects of the powergrid.

    Deriving from this class allows to perform the display of the powergrid.

    **NB** this class only performs the computation of the position, but does not display anything.

    Attributes
    -----------

    observation_space: :class:`grid2op.Observation.HelperObservation`
        The observation space used.

    """

    def __init__(
        self,
        observation_space,
        substation_layout=None,
        radius_sub=20.0,
        load_prod_dist=70.0,
        bus_radius=6.0,
    ):

        warnings.warn(
            "This whole class has been deprecated. Use `grid2op.PlotGrid module instead`",
            category=DeprecationWarning,
        )
        if substation_layout is None:
            if observation_space.grid_layout is None:
                # if no layout is provided, and observation_space has no layout, then it fails
                raise PlotError(
                    "Impossible to use plotting abilities without specifying a layout (coordinates) "
                    "of the substations."
                )

            # if no layout is provided, use the one in the observation_space
            substation_layout = []
            for el in observation_space.name_sub:
                substation_layout.append(observation_space.grid_layout[el])

        if len(substation_layout) != observation_space.n_sub:
            raise PlotError(
                "You provided a layout with {} elements while there are {} substations on the powergrid. "
                "Your layout is invalid".format(
                    len(substation_layout), observation_space.n_sub
                )
            )
        GridObjects.__init__(self)
        self.init_grid(observation_space)

        self.observation_space = observation_space
        self._layout = {}
        self._layout["substations"] = self._get_sub_layout(substation_layout)

        self.radius_sub = radius_sub
        self.load_prod_dist = load_prod_dist  # distance between load and generator to the center of the substation
        self.bus_radius = bus_radius

        self.subs_elements = [None for _ in self.observation_space.sub_info]

        # get the element in each substation
        for sub_id in range(self.observation_space.sub_info.shape[0]):
            this_sub = {}
            objs = self.observation_space.get_obj_connect_to(substation_id=sub_id)

            for c_id in objs["loads_id"]:
                c_nm = self._get_load_name(sub_id, c_id)
                this_load = {}
                this_load["type"] = "load"
                this_load["sub_pos"] = self.observation_space.load_to_sub_pos[c_id]
                this_sub[c_nm] = this_load

            for g_id in objs["generators_id"]:
                g_nm = self._get_gen_name(sub_id, g_id)
                this_gen = {}
                this_gen["type"] = "gen"
                this_gen["sub_pos"] = self.observation_space.gen_to_sub_pos[g_id]
                this_sub[g_nm] = this_gen

            for lor_id in objs["lines_or_id"]:
                ext_id = self.observation_space.line_ex_to_subid[lor_id]
                l_nm = self._get_line_name(sub_id, ext_id, lor_id)
                this_line = {}
                this_line["type"] = "line"
                this_line["sub_pos"] = self.observation_space.line_or_to_sub_pos[lor_id]
                this_sub[l_nm] = this_line

            for lex_id in objs["lines_ex_id"]:
                or_id = self.observation_space.line_or_to_subid[lex_id]
                l_nm = self._get_line_name(or_id, sub_id, lex_id)
                this_line = {}
                this_line["type"] = "line"
                this_line["sub_pos"] = self.observation_space.line_ex_to_sub_pos[lex_id]
                this_sub[l_nm] = this_line
            self.subs_elements[sub_id] = this_sub
        self._compute_layout()

        self.col_line = None
        self.col_sub = None
        self.col_load = None
        self.col_gen = None
        self.default_color = None

    def plot_layout(self, fig=None, reward=None, done=None, timestamp=None):
        """
        .. warning:: /!\\\\ This module is deprecated /!\\\\

            Prefer using the module `grid2op.PlotGrid

        This function plot the layout of the grid, as well as the object. You will see the name of each elements and
        their id.
        """
        fig = self.init_fig(fig, reward, done, timestamp)
        # draw powerline
        lines = self._draw_powerlines(fig)
        # draw substation
        subs = self._draw_subs(fig)
        # draw loads
        loads = self._draw_loads(fig)
        # draw gens
        gens = self._draw_gens(fig)
        self._post_process_obs(
            fig,
            reward=None,
            done=None,
            timestamp=None,
            subs=subs,
            lines=lines,
            loads=loads,
            gens=gens,
            topos=[],
        )
        return fig

    def plot_info(
        self,
        fig=None,
        line_info=None,
        load_info=None,
        gen_info=None,
        sub_info=None,
        colormap=None,
        unit=None,
    ):

        """
        .. warning:: /!\\\\ This module is deprecated /!\\\\

            Prefer using the module `grid2op.PlotGrid

        Plot some information on the powergrid. For now, only numeric data are supported.

        Parameters
        ----------
        line_info: ``list``
            information to be displayed in the powerlines, in place of their name and id (for example their
            thermal limit) [must have the same size as the number of powerlines and convertible to float]

        load_info: ``list``
            information to display in the generators, in place of their name and id
            [must have the same size as the number of loads and convertible to float]

        gen_info: ``list``
            information to display in the generators, in place of their name and id (for example their pmax)
            [must have the same size as the number of generators and convertible to float]

        sub_info: ``list``
            information to display in the substation, in place of their name and id (for example the number of
            different topologies possible at this substation) [must have the same size as the number of substations,
            and convertible to float]

        colormap: ``str``
            If not None, one of "line", "load", "gen" or "sub". If None, default colors will be used for each
            elements (default color is the coloring of
            If not None, all elements will be black, and the selected element will be highlighted.

        fig: ``matplotlib figure``
            The figure on which to draw. It is created by the method if ``None``.

        unit: ``str``, optional
            The unit in which the data are provided. For example, if you provide in `line_info` some data in mega-watt
            (MW) you can add `unit="MW"` to have the unit display on the screen.

        """
        fig = self.init_fig(fig, reward=None, done=None, timestamp=None)

        # draw powerline
        unit_line = None
        if line_info is not None:
            unit_line = unit
            if len(line_info) != self.n_line:
                raise PlotError(
                    "Impossible to display these information on the powerlines: there are {} elements"
                    "provided while {} powerlines on this grid".format(
                        len(line_info), self.n_line
                    )
                )
            line_info = np.array(line_info).astype(np.float)
        line_info = [line_info, line_info, line_info]
        lines = self._draw_powerlines(
            fig, vals=line_info, colormap=colormap, unit=unit_line
        )

        # draw substation
        unit_sub = None
        if sub_info is not None:
            unit_sub = unit
            if len(sub_info) != self.n_sub:
                raise PlotError(
                    "Impossible to display these information on the substations: there are {} elements"
                    "provided while {} substations on this grid".format(
                        len(sub_info), self.n_sub
                    )
                )
            sub_info = np.array(sub_info).astype(np.float)
        subs = self._draw_subs(fig, vals=sub_info, colormap=colormap, unit=unit_sub)

        # draw loads
        unit_load = None
        if load_info is not None:
            unit_load = unit
            if len(load_info) != self.n_load:
                raise PlotError(
                    "Impossible to display these information on the loads: there are {} elements"
                    "provided while {} loads on this grid".format(
                        len(load_info), self.n_load
                    )
                )
            load_info = np.array(load_info).astype(np.float)
        loads = self._draw_loads(fig, vals=load_info, colormap=colormap, unit=unit_load)

        # draw gens
        unit_gen = None
        if gen_info is not None:
            unit_gen = unit
            if len(gen_info) != self.n_gen:
                raise PlotError(
                    "Impossible to display these information on the generators: there are {} elements"
                    "provided while {} generators on this grid".format(
                        len(gen_info), self.n_gen
                    )
                )
            gen_info = np.array(gen_info).astype(np.float)
        gens = self._draw_gens(fig, vals=gen_info, colormap=colormap, unit=unit_gen)

        self._post_process_obs(
            fig,
            reward=None,
            done=None,
            timestamp=None,
            subs=subs,
            lines=lines,
            loads=loads,
            gens=gens,
            topos=[],
        )
        return fig

    def plot_obs(
        self,
        observation,
        fig=None,
        reward=None,
        done=None,
        timestamp=None,
        line_info="rho",
        load_info="p",
        gen_info="p",
        colormap="line",
    ):
        """
        .. warning:: /!\\\\ This module is deprecated /!\\\\

            Prefer using the module `grid2op.PlotGrid

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

        line_info: ``str``
            One of "rho", "a", or "p" or "v" the information that will be plotted on the powerline By default "rho".
            All flow are taken "origin" side.

        load_info: ``str``
            One of "p" or "v" the information displayed on the load (défault to "p").

        gen_info: ``str``
            One of "p" or "v" the information displayed on the generators (default to "p").

        Returns
        -------
        res: :class:`plotly.graph_objects.Figure`
            The figure updated with the data from the new observation.
        """
        fig = self.init_fig(fig, reward, done, timestamp)

        # draw substation
        subs = self._draw_subs(fig=fig, vals=[None for el in range(self.n_sub)])

        # draw powerlines
        if line_info == "rho":
            line_vals = [observation.rho]
            line_units = "%"
        elif line_info == "a":
            line_vals = [observation.a_or]
            line_units = "A"
        elif line_info == "p":
            line_vals = [observation.p_or]
            line_units = "MW"
        elif line_info == "v":
            line_vals = [observation.v_or]
            line_units = "kV"
        else:
            raise PlotError(
                'Impossible to plot value "{}" for line. Possible values are "rho", "p", "v" and "a".'
            )
        line_vals.append(observation.line_status)
        line_vals.append(observation.p_or)
        lines = self._draw_powerlines(
            fig, vals=line_vals, unit=line_units, colormap=colormap
        )

        # draw the loads
        if load_info == "p":
            loads_vals = -observation.load_p
            load_units = "MW"
        elif load_info == "v":
            loads_vals = observation.load_v
            load_units = "kV"
        else:
            raise PlotError(
                'Impossible to plot value "{}" for load. Possible values are "p" and "v".'
            )
        loads = self._draw_loads(
            fig, vals=loads_vals, unit=load_units, colormap=colormap
        )

        # draw the generators
        if gen_info == "p":
            gen_vals = observation.prod_p
            gen_units = "MW"
        elif gen_info == "v":
            gen_vals = observation.prod_v
            gen_units = "kV"
        else:
            raise PlotError(
                'Impossible to plot value "{}" for generators. Possible values are "p" and "v".'
            )
        gens = self._draw_gens(fig, vals=gen_vals, unit=gen_units, colormap=colormap)
        # draw the topologies
        topos = self._draw_topos(fig=fig, observation=observation)
        self._post_process_obs(
            fig, reward, done, timestamp, subs, lines, loads, gens, topos
        )
        return fig

    def _get_sub_layout(self, init_layout):
        return init_layout

    def _get_line_name(self, subor_id, sub_ex_id, line_id):
        l_nm = "l_{}_{}_{}".format(subor_id, sub_ex_id, line_id)
        return l_nm

    def _get_load_name(self, sub_id, c_id):
        c_nm = "load_{}_{}".format(sub_id, c_id)
        return c_nm

    def _get_gen_name(self, sub_id, g_id):
        p_nm = "gen_{}_{}".format(sub_id, g_id)
        return p_nm

    def _compute_layout(self):
        """

        .. warning:: /!\\\\ This module is deprecated /!\\\\

            Prefer using the module `grid2op.PlotGrid

        Compute the position of each of the objects.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The observation used to know which object belong where.

        Returns
        -------

        """
        self._layout["line"] = {}

        # assign powerline coordinates
        for line_id in range(self.n_line):
            if line_id not in self._layout["line"]:
                # state = observation.state_of(line_id=line_id)
                sub_or_id = self.line_or_to_subid[line_id]  # state["origin"]["sub_id"]
                sub_ex_id = self.line_ex_to_subid[
                    line_id
                ]  # state["extremity"]["sub_id"]
                pos_or = self._layout["substations"][sub_or_id]
                pos_ex = self._layout["substations"][sub_ex_id]

                # make sure the powerline are connected to the circle of the substation and not to the center of it
                z_or_tmp = pos_or[0] + 1j * pos_or[1]
                z_ex_tmp = pos_ex[0] + 1j * pos_ex[1]

                module_or = cmath.phase(z_ex_tmp - z_or_tmp)
                module_ex = cmath.phase(-(z_ex_tmp - z_or_tmp))

                # check parrallel lines:
                # for now it works only if there are 2 parrallel lines. The idea is to add / withdraw
                # 10° for each module in this case.
                # TODO draw line but not straight line in this case, this looks ugly for now :-/
                deg_parrallel = 25
                tmp_parrallel = self.observation_space.get_lines_id(
                    from_=sub_or_id, to_=sub_ex_id
                )
                if len(tmp_parrallel) > 1:
                    if line_id == tmp_parrallel[0]:
                        module_or += deg_parrallel / 360 * 2 * cmath.pi
                        module_ex -= deg_parrallel / 360 * 2 * cmath.pi
                    else:
                        module_or -= deg_parrallel / 360 * 2 * cmath.pi
                        module_ex += deg_parrallel / 360 * 2 * cmath.pi

                z_or = z_or_tmp + self.radius_sub * cmath.exp(module_or * 1j)
                z_ex = z_ex_tmp + self.radius_sub * cmath.exp(module_ex * 1j)
                pos_or = z_or.real, z_or.imag
                pos_ex = z_ex.real, z_ex.imag
                self._layout["line"][line_id] = sub_or_id, sub_ex_id
                # TODO here get proper name
                l_nm = self._get_line_name(sub_or_id, sub_ex_id, line_id)

                self.subs_elements[sub_or_id][l_nm]["pos"] = pos_or
                self.subs_elements[sub_or_id][l_nm]["z"] = z_or
                self.subs_elements[sub_ex_id][l_nm]["pos"] = pos_ex
                self.subs_elements[sub_ex_id][l_nm]["z"] = z_ex

        # assign loads and generators coordinates
        # this is done by first computing the "optimal" placement if there were only substation (so splitting equally
        # the objects around the circle) and then remove the closest position that are taken by the powerlines.
        for sub_id, elements in enumerate(self.subs_elements):
            nb_el = len(elements)

            # equally split
            pos_sub = self._layout["substations"][sub_id]
            z_sub = pos_sub[0] + 1j * pos_sub[1]
            pos_possible = [
                self.radius_sub * cmath.exp(1j * 2 * cmath.pi * i / nb_el) + z_sub
                for i in range(nb_el)
            ]

            # remove powerlines (already assigned)
            for el_nm, dict_el in elements.items():
                if dict_el["type"] == "line":
                    z = dict_el["z"]
                    closest = np.argmin([abs(pos - z) ** 2 for pos in pos_possible])
                    pos_possible = [
                        el for i, el in enumerate(pos_possible) if i != closest
                    ]

            i = 0
            # now assign load and generator
            for el_nm, dict_el in elements.items():
                if dict_el["type"] != "line":
                    dict_el["pos"] = (pos_possible[i].real, pos_possible[i].imag)
                    dict_el["z"] = pos_possible[i]
                    i += 1

        self._layout["load"] = {}
        for c_id in range(self.n_load):
            # state = observation.state_of(load_id=c_id)
            # sub_id = state["sub_id"]
            sub_id = self.load_to_subid[c_id]
            self._layout["load"][c_id] = sub_id

        self._layout["gen"] = {}
        for g_id in range(self.n_gen):
            # state = observation.state_of(gen_id=g_id)
            # sub_id = state["sub_id"]
            sub_id = self.gen_to_subid[g_id]
            self._layout["gen"][g_id] = sub_id

    def _get_line_coord(self, line_id):
        sub_or_id, sub_ex_id = self._layout["line"][line_id]
        l_nm = self._get_line_name(sub_or_id, sub_ex_id, line_id)
        pos_or = self.subs_elements[sub_or_id][l_nm]["pos"]
        pos_ex = self.subs_elements[sub_ex_id][l_nm]["pos"]
        return pos_or, pos_ex

    def _get_load_coord(self, load_id):
        sub_id = self._layout["load"][load_id]
        c_nm = self._get_load_name(sub_id, load_id)

        if not "elements_display" in self.subs_elements[sub_id][c_nm]:
            pos_load_sub = self.subs_elements[sub_id][c_nm]["pos"]
            pos_center_sub = self._layout["substations"][sub_id]

            z_sub = pos_center_sub[0] + 1j * pos_center_sub[1]
            theta = cmath.phase((self.subs_elements[sub_id][c_nm]["z"] - z_sub))
            pos_load = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

            # position of the end of the line connecting the object to the substation
            pos_end_line = pos_load - cmath.exp(1j * theta) * 20
            how_center = self._get_position(theta)
            tmp_dict = {
                "pos_end_line": pos_end_line,
                "pos_load_sub": pos_load_sub,
                "pos_load": pos_load,
                "how_center": how_center,
            }
            self.subs_elements[sub_id][c_nm]["elements_display"] = tmp_dict
        else:
            dict_element = self.subs_elements[sub_id][c_nm]["elements_display"]
            pos_end_line = dict_element["pos_end_line"]
            pos_load_sub = dict_element["pos_load_sub"]
            pos_load = dict_element["pos_load"]
            how_center = dict_element["how_center"]

        return pos_end_line, pos_load_sub, pos_load, how_center

    def _get_gen_coord(self, gen_id):
        sub_id = self._layout["gen"][gen_id]
        c_nm = self._get_gen_name(sub_id, gen_id)

        if not "elements_display" in self.subs_elements[sub_id][c_nm]:
            pos_gen_sub = self.subs_elements[sub_id][c_nm]["pos"]
            pos_center_sub = self._layout["substations"][sub_id]

            z_sub = pos_center_sub[0] + 1j * pos_center_sub[1]
            theta = cmath.phase((self.subs_elements[sub_id][c_nm]["z"] - z_sub))
            pos_gen = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

            # position of the end of the line connecting the object to the substation
            pos_end_line = pos_gen - cmath.exp(1j * theta) * 20
            how_center = self._get_position(theta)
            tmp_dict = {
                "pos_end_line": pos_end_line,
                "pos_gen_sub": pos_gen_sub,
                "pos_gen": pos_gen,
                "how_center": how_center,
            }
            self.subs_elements[sub_id][c_nm]["elements_display"] = tmp_dict
        else:
            dict_element = self.subs_elements[sub_id][c_nm]["elements_display"]
            pos_end_line = dict_element["pos_end_line"]
            pos_gen_sub = dict_element["pos_gen_sub"]
            pos_gen = dict_element["pos_gen"]
            how_center = dict_element["how_center"]

        return pos_end_line, pos_gen_sub, pos_gen, how_center

    def _get_topo_coord(self, sub_id, observation, elements):
        pos_center_sub = self._layout["substations"][sub_id]
        z_sub = pos_center_sub[0] + 1j * pos_center_sub[1]

        tmp = observation.state_of(substation_id=sub_id)
        if tmp["nb_bus"] == 1:
            # not to overload the plot, if everything is at the same bus, i don't plot it
            return [], []
        # I have at least 2 buses

        # I compute the position of each elements
        bus_vect = tmp["topo_vect"]

        # i am not supposed to have more than 2 buses
        buses_z = [None, None]  # center of the different buses
        nb_co = [0, 0]  # center of the different buses

        # the position of a bus is for now the average of all the elements in there
        for el_nm, dict_el in elements.items():
            this_el_bus = bus_vect[dict_el["sub_pos"]] - 1
            if this_el_bus >= 0:
                nb_co[this_el_bus] += 1
                if buses_z[this_el_bus] is None:
                    buses_z[this_el_bus] = dict_el["z"]
                else:
                    buses_z[this_el_bus] += dict_el["z"]

        #
        buses_z = [el / nb for el, nb in zip(buses_z, nb_co)]
        theta_z = [cmath.phase((el - z_sub)) for el in buses_z]

        # try to have nodes "in opposition" to one another
        NN = np.array(nb_co) / nb_co.sum()
        diff_theta = theta_z[0] - theta_z[1]
        # alpha = cmath.pi + diff_theta
        alpha = -cmath.pi + diff_theta
        alpha = math.fmod(alpha, 2 * cmath.pi)
        theta_z = [theta_z[0] - alpha * NN[1], theta_z[1] + alpha * NN[0]]

        # buses_z = [z_sub + (self.radius_sub - self.bus_radius) * 0.75 * cmath.exp(1j * theta) for theta in theta_z]
        buses_z = [
            z_sub + (self.radius_sub - self.bus_radius) * 0.6 * cmath.exp(1j * theta)
            for theta in theta_z
        ]
        return buses_z, bus_vect

    @staticmethod
    def _get_position(theta):
        quarter_pi = cmath.pi / 4
        half_pi = cmath.pi / 2.0
        if theta >= -quarter_pi and theta < quarter_pi:
            res = "center|left"
        elif theta >= quarter_pi and theta < quarter_pi + half_pi:
            res = "up|center"
        elif theta >= quarter_pi + half_pi and theta < quarter_pi + 2.0 * half_pi:
            res = "center|right"
        else:
            res = "down|center"
        return res

    def _get_text_unit(self, number, unit):
        if number is not None:
            if isinstance(number, float) or isinstance(number, np.float):
                if np.isfinite(number):
                    if unit == "%":
                        number *= 100.0
                    number = "{:.1f}".format(number)
                else:
                    return None

            if unit is not None:
                txt_ = "{}{}".format(number, unit)
            else:
                txt_ = number
        else:
            return None
        return txt_

    def _draw_subs(self, fig=None, vals=None, colormap=None, unit=None):
        subs = []
        colormap_ = lambda x: self.col_sub
        texts = None
        if vals is not None:
            texts = [self._get_text_unit(val, unit) for val in vals]

        if colormap is not None:
            if colormap == "sub":
                # normalize value for the color map
                vals = self._get_vals(vals)

        if texts is not None:
            vals = [float(text if text is not None else 0.0) for text in texts]

        for sub_id, center in enumerate(self._layout["substations"]):
            if texts is None:
                txt_ = "{}\nid: {}".format(self.name_sub[sub_id], sub_id)
                this_col = colormap_("")
            else:
                txt_ = texts[sub_id]
                if colormap == "sub":
                    this_col = self._get_sub_color_map(vals[sub_id])
                else:
                    this_col = self.default_color
            subs.append(self._draw_subs_one_sub(fig, sub_id, center, this_col, txt_))
        return subs

    def get_sub_color_map(self):
        return None

    def _draw_subs_one_sub(self, fig, sub_id, center, this_col, text):
        return None

    def _draw_powerlines(self, fig=None, vals=None, colormap=None, unit=None):
        lines = []

        colormap_ = lambda x: self.col_line
        texts = None
        if vals is not None:
            vals_0 = vals[0]
            texts = [self._get_text_unit(val, unit) for val in vals[0]]

        if colormap is not None:
            if colormap == "line" and unit != "%":
                # normalize the value for the color map
                vals_0 = self._get_vals(vals[0])

        for line_id in range(self.n_line):
            pos_or, pos_ex, *_ = self._get_line_coord(line_id)

            if texts is None:
                txt_ = "{}\nid: {}".format(self.name_line[line_id], line_id)
                this_col = colormap_("")
            else:
                txt_ = texts[line_id]
                if colormap == "line":
                    this_col = self._get_line_color_map(vals_0[line_id])
                else:
                    this_col = self.default_color

            if vals is not None:
                value = vals_0[line_id]
                status = vals[1][line_id]
                por = vals[2][line_id]
            else:
                value = 0.0
                status = True
                por = 1.0

            if por is None:
                por = 1.0
            if status is None:
                status = True

            if not status:
                this_col = self.default_color
            lines.append(
                self._draw_powerlines_one_powerline(
                    fig,
                    line_id,
                    pos_or,
                    pos_ex,
                    status,
                    value,
                    txt_,
                    por >= 0.0,
                    this_col,
                )
            )
        return lines

    def _draw_powerlines_one_powerline(
        self, fig, l_id, pos_or, pos_ex, status, value, txt_, or_to_ex, this_col
    ):
        return None

    def _draw_loads(self, fig=None, vals=None, colormap=None, unit=None):
        loads = []

        colormap_ = lambda x: self.col_load
        texts = None
        if vals is not None:
            texts = [self._get_text_unit(val, unit) for val in vals]

        if colormap is not None:
            if colormap == "load":
                # normalized the value for the color map
                vals = self._get_vals(vals)

        for c_id in range(self.n_load):
            pos_end_line, pos_load_sub, pos_load, how_center = self._get_load_coord(
                c_id
            )
            if texts is None:
                txt_ = "{}\nid: {}".format(self.name_load[c_id], c_id)
                this_col = colormap_("")
            else:
                txt_ = texts[c_id]
                if colormap == "load":
                    this_col = self._get_load_color_map(vals[c_id])
                else:
                    this_col = self.default_color

            loads.append(
                self._draw_loads_one_load(
                    fig,
                    c_id,
                    pos_load,
                    txt_,
                    pos_end_line,
                    pos_load_sub,
                    how_center,
                    this_col,
                )
            )
        return loads

    def _draw_loads_one_load(
        self,
        fig,
        l_id,
        pos_load,
        txt_,
        pos_end_line,
        pos_load_sub,
        how_center,
        this_col,
    ):
        return None

    def _get_sub_color_map(self, normalized_val):
        return self._get_default_cmap(normalized_val)

    def _get_load_color_map(self, normalized_val):
        return self._get_default_cmap(normalized_val)

    def _get_gen_color_map(self, normalized_val):
        return self._get_default_cmap(normalized_val)

    def _get_line_color_map(self, normalized_val):
        return self._get_default_cmap(normalized_val)

    def _get_default_cmap(self, normalized_val):
        return self.default_color

    def _get_vals(self, vals):
        vals = copy.deepcopy(vals)
        min_ = np.min(vals)
        max_ = np.max(vals)
        vals -= min_
        vals /= max_ - min_ + 1e-5
        # now vals is between 0 and 1, i push it toward 1 a bit to better see it
        vals += 0.5
        vals /= 1.5
        return vals

    def _draw_gens(self, fig=None, vals=None, colormap=None, unit=None):
        gens = []

        colormap_ = lambda x: self.col_gen
        texts = None
        if vals is not None:
            texts = [self._get_text_unit(val, unit) for val in vals]

        if colormap is not None:
            if colormap == "gen":
                # normalized the value for plot
                vals = self._get_vals(vals)

        for g_id in range(self.n_gen):
            pos_end_line, pos_gen_sub, pos_gen, how_center = self._get_gen_coord(g_id)
            if texts is None:
                txt_ = "{}\nid: {}".format(self.name_gen[g_id], g_id)
                this_col = colormap_("")
            else:
                txt_ = texts[g_id]
                if colormap == "gen":
                    this_col = self._get_gen_color_map(vals[g_id])
                else:
                    this_col = self.default_color
            gens.append(
                self._draw_gens_one_gen(
                    fig,
                    g_id,
                    pos_gen,
                    txt_,
                    pos_end_line,
                    pos_gen_sub,
                    how_center,
                    this_col,
                )
            )
        return gens

    def _draw_gens_one_gen(
        self, fig, g_id, pos_gen, txt_, pos_end_line, pos_gen_sub, how_center, this_col
    ):
        return None

    def _draw_topos(self, observation, fig):
        res_topo = []
        for sub_id, elements in enumerate(self.subs_elements):
            buses_z, bus_vect = self._get_topo_coord(sub_id, observation, elements)

            if not buses_z:
                # I don't plot details of substations with 1 bus for better quality
                continue
            res_topo += self._draw_topos_one_sub(
                fig, sub_id, buses_z, elements, bus_vect
            )
        return res_topo

    def _draw_topos_one_sub(self, fig, sub_id, buses_z, elements, bus_vect):
        return [None]

    def _post_process_obs(
        self, fig, reward, done, timestamp, subs, lines, loads, gens, topos
    ):
        pass

    def init_fig(self, fig, reward, done, timestamp):
        pass

    ## DEPRECATED FUNCTIONS
    def plot_observation(
        self, observation, fig=None, line_info="rho", load_info="p", gen_info="p"
    ):
        """

        .. warning:: /!\\\\ This module is deprecated /!\\\\

            Prefer using the module `grid2op.PlotGrid

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The observation to plot

        fig: ``plotly figure``
            will be created if ``None``

        line_info: ``str``
            One of "rho", "a", or "p" the information that will be plotted on the powerline By default "rho".

        load_info: ``str``
            One of "p" or "v" the information displayed on the load (défault to "p").

        gen_info: ``str``
            One of "p" or "v" the information displayed on the generators (default to "p").

        Returns
        -------
        res: ``plotly figure``
            The resulting figure.
        """
        warnings.warn(
            '"plot_observation" method will be deprecated in future version. '
            'Please use "plot_obs" instead.',
            category=PendingDeprecationWarning,
        )

        res = self.plot_obs(
            observation,
            fig=fig,
            line_info=line_info,
            load_info=load_info,
            gen_info=gen_info,
        )
        return res

    def get_plot_observation(
        self, observation, fig=None, line_info="rho", load_info="p", gen_info="p"
    ):
        """

        .. warning:: /!\\\\ This module is deprecated /!\\\\

            Prefer using the module `grid2op.PlotGrid

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The observation to plot

        fig: ``plotly figure``
            will be created if ``None``

        line_info: ``str``
            One of "rho", "a", or "p" the information that will be plotted on the powerline By default "rho".

        load_info: ``str``
            One of "p" or "v" the information displayed on the load (défault to "p").

        gen_info: ``str``
            One of "p" or "v" the information displayed on the generators (default to "p").

        Returns
        -------
        res: ``plotly figure``
            The resulting figure.
        """
        warnings.warn(
            '"get_plot_observation" method will be deprecated in future version. '
            'Please use "plot_obs" instead.',
            category=PendingDeprecationWarning,
        )

        res = self.plot_obs(
            observation,
            fig=fig,
            line_info=line_info,
            load_info=load_info,
            gen_info=gen_info,
        )
        return res
