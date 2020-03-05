"""
This module is the base module for all graphical representation of the :class:`grid2op.Observation.Observation`.

It allows, from the layout of the graph of the powergrid (*eg* the coordinates of each substation) the position of
each objects (powerline ends, loads and generators) and the position of the buses in case of "node splitting" (when
a substations is split into independent electrical buses).

"""
import cmath
import math
import numpy as np
import pdb

try:
    from .Space import GridObjects
    from .Exceptions import PlotError
except:
    from Space import GridObjects
    from Exceptions import PlotError


class BasePlot(GridObjects):
    """
    Utility class that allows to compute the position of the objects of the powergrid.

    Deriving from this class allows to perform the display of the powergrid.

    **NB** this class only performs the computation of the position, but does not display anything.

    Attributes
    -----------

    observation_space: :class:`grid2op.Observation.HelperObservation`
        The observation space used.

    """
    def __init__(self,
                 substation_layout,
                 observation_space,
                 radius_sub=20.,
                 load_prod_dist=70.,
                 bus_radius=6.):
        if substation_layout is None:
            raise PlotError("Impossible to use plotting abilities without specifying a layout (coordinates) "
                                   "of the substations.")

        if len(substation_layout) != observation_space.n_sub:
            raise PlotError("You provided a layout with {} elements while there are {} substations on the powergrid. "
                            "Your layout is invalid".format(len(substation_layout), observation_space.n_sub))
        GridObjects.__init__(self)
        self.init_grid(observation_space)

        self.observation_space = observation_space
        self._layout = {}
        self._layout["substations"] = self._get_sub_layout(substation_layout)

        self.radius_sub = radius_sub
        self.load_prod_dist = load_prod_dist # distance between load and generator to the center of the substation
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

    def _get_sub_layout(self, init_layout):
        return init_layout

    def _get_line_name(self, subor_id, sub_ex_id, line_id):
        l_nm = 'l_{}_{}_{}'.format(subor_id, sub_ex_id, line_id)
        return l_nm

    def _get_load_name(self, sub_id, c_id):
        c_nm = "load_{}_{}".format(sub_id, c_id)
        return c_nm

    def _get_gen_name(self, sub_id, g_id):
        p_nm = 'gen_{}_{}'.format(sub_id, g_id)
        return p_nm

    def _compute_layout(self):
        """
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
                sub_ex_id = self.line_ex_to_subid[line_id]  # state["extremity"]["sub_id"]
                pos_or = self._layout["substations"][sub_or_id]
                pos_ex = self._layout["substations"][sub_ex_id]

                # make sure the powerline are connected to the circle of the substation and not to the center of it
                z_or_tmp = pos_or[0] + 1j * pos_or[1]
                z_ex_tmp = pos_ex[0] + 1j * pos_ex[1]

                module_or = cmath.phase(z_ex_tmp - z_or_tmp)
                module_ex = cmath.phase(- (z_ex_tmp - z_or_tmp))

                # check parrallel lines:
                # for now it works only if there are 2 parrallel lines. The idea is to add / withdraw
                # 10° for each module in this case.
                # TODO draw line but not straight line in this case, this looks ugly for now :-/
                deg_parrallel = 25
                tmp_parrallel = self.observation_space.get_lines_id(from_=sub_or_id, to_=sub_ex_id)
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
            pos_possible = [self.radius_sub * cmath.exp(1j * 2 * cmath.pi * i / nb_el) + z_sub
                            for i in range(nb_el)]

            # remove powerlines (already assigned)
            for el_nm, dict_el in elements.items():
                if dict_el["type"] == "line":
                    z = dict_el["z"]
                    closest = np.argmin([abs(pos - z)**2 for pos in pos_possible])
                    pos_possible = [el for i, el in enumerate(pos_possible) if i != closest]

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

    def _draw_subs(self, observation):
        res = []
        for i, el in enumerate(self._layout["substations"]):
            res.append(self._draw_sub(center=el))
        return res

    def _draw_sub(self, center):
        pass

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

            z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])
            theta = cmath.phase((self.subs_elements[sub_id][c_nm]["z"] - z_sub))
            pos_load = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

            # position of the end of the line connecting the object to the substation
            pos_end_line = pos_load - cmath.exp(1j * theta) * 20
            how_center = self._get_position(theta)
            tmp_dict = {"pos_end_line": pos_end_line,
                        "pos_load_sub": pos_load_sub,
                        "pos_load": pos_load,
                        "how_center": how_center}
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

            z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])
            theta = cmath.phase((self.subs_elements[sub_id][c_nm]["z"] - z_sub))
            pos_gen = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

            # position of the end of the line connecting the object to the substation
            pos_end_line = pos_gen - cmath.exp(1j * theta) * 20
            how_center = self._get_position(theta)
            tmp_dict = {"pos_end_line": pos_end_line,
                        "pos_gen_sub": pos_gen_sub,
                        "pos_gen": pos_gen,
                        "how_center": how_center}
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
        z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])

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
        NN = np.array(nb_co) / np.sum(nb_co)
        diff_theta = theta_z[0] - theta_z[1]
        # alpha = cmath.pi + diff_theta
        alpha = -cmath.pi + diff_theta
        alpha = math.fmod(alpha, 2*cmath.pi)
        theta_z = [theta_z[0] - alpha * NN[1], theta_z[1] + alpha * NN[0]]

        # buses_z = [z_sub + (self.radius_sub - self.bus_radius) * 0.75 * cmath.exp(1j * theta) for theta in theta_z]
        buses_z = [z_sub + (self.radius_sub - self.bus_radius) * 0.6 * cmath.exp(1j * theta) for theta in theta_z]
        return buses_z, bus_vect

    @staticmethod
    def _get_position(theta):
        quarter_pi = cmath.pi / 4
        half_pi = cmath.pi / 2.

        if theta >= -quarter_pi and theta < quarter_pi:
            res = "center|left"
        elif theta >= quarter_pi and theta < quarter_pi + half_pi:
            res = "up|center"
        elif theta >= quarter_pi + half_pi and theta < quarter_pi + 2. * half_pi:
            res = "center|right"
        else:
            res = "down|center"

        return res
