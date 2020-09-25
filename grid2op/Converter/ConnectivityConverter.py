# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np

from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_int, dt_bool


class ConnectivityConverter(Converter):
    """
    In this converter, you have as many output as pairs of object that can be connected, and your model is asked
    to output 0 if he wants these elements disconnected and 1 if he wants them otherwise.

    This type of modeling is rather hard to "get working" the first time, especially because some "conflict" might
    appear. For example, consider three objects (line for example) on a given substation. You can chose to "connect
    A and B", connect "B and C" but "**not connect** A and C" in this case you need an algorithm to disambuate your
    action.

    It can not yet be converted to / from gym space. If this feature is interesting for you, you can
    reply to the issue posted at https://github.com/rte-france/Grid2Op/issues/16

    **NB** compare to :class:`IdToAct` this converter allows for a smaller size. If you have N elements connected at
    a substation, you end up with `N*(N-1)/2` different action. Compare to IdToAct though, it is expected that your
    algorithm produces more than 1 output.

    **VERY IMPORTANT** : for this converter to work, it needs to remember the previous state of the grid, so you
    absolutely need to call its method :func:`ConnectivityConverter.convert_obs` a each observation.
    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = ConnectivityConverter.init_grid(action_space)
        self.subs_ids = np.array([], dtype=dt_int)
        self.obj_type = []
        self.pos_topo = np.array([], dtype=dt_int)

        self.n = 1
        self.last_obs = None
        self.max_sub_changed = self.n_sub
        self.last_disagreement = None

    def init_converter(self, all_actions=None, **kwargs):
        # compute all pairs of elements that can be connected together
        self.pos_topo = []
        self.subs_ids = []
        for sub_id, nb_element in enumerate(self.sub_info):
            if nb_element < 4:
                continue

            c_id = np.where(self.load_to_subid == sub_id)[0]
            g_id = np.where(self.gen_to_subid == sub_id)[0]
            lor_id = np.where(self.line_or_to_subid == sub_id)[0]
            lex_id = np.where(self.line_ex_to_subid == sub_id)[0]

            c_pos = self.load_to_sub_pos[self.load_to_subid == sub_id]
            g_pos = self.gen_to_sub_pos[self.gen_to_subid == sub_id]
            lor_pos = self.line_or_to_sub_pos[self.line_or_to_subid == sub_id]
            lex_pos = self.line_ex_to_sub_pos[self.line_ex_to_subid == sub_id]

            my_types = []
            pos_topo = []
            next_load_ = 0
            next_gen_ = 0
            next_lor_ = 0
            next_lex_ = 0
            next_load = c_id[next_load_] if c_id.shape[0] > 0 else None
            next_gen = g_id[next_gen_] if g_id.shape[0] > 0 else None
            next_lor = lor_id[next_lor_] if lor_id.shape[0] > 0 else None
            next_lex = lex_id[next_lex_] if lex_id.shape[0] > 0 else None
            for id_i in range(nb_element):
                type_i, id_obj_i = self._get_id_from_obj(id_i,
                                                         c_pos, g_pos, lor_pos, lex_pos,
                                                         next_load, next_gen, next_lor, next_lex)
                if type_i == "load":
                    next_load_ += 1
                    next_load = c_id[next_load_] if c_id.shape[0] > next_load_ else None
                elif type_i == "gen":
                    next_gen_ += 1
                    next_gen = g_id[next_gen_] if g_id.shape[0] > next_gen_ else None
                elif type_i == "line_or":
                    next_lor_ += 1
                    next_lor = lor_id[next_lor_] if lor_id.shape[0] > next_lor_ else None
                elif type_i == "line_ex":
                    next_lex_ += 1
                    next_lex = lex_id[next_lex_] if lex_id.shape[0] > next_lex_ else None
                my_types.append((type_i, id_obj_i))
                pos_topo.append(self._get_pos_topo(type_i, id_obj_i))

            for id_i in range(nb_element):
                id_i_ = my_types[id_i]
                pos_topo_i = pos_topo[id_i]
                for id_j in range(id_i+1, nb_element):
                    id_j_ = my_types[id_j]
                    pos_topo_j = pos_topo[id_j]
                    self.obj_type.append((sub_id, id_i_, id_j_))
                    self.pos_topo.append((pos_topo_i, pos_topo_j))
                    self.subs_ids.append(sub_id)

        self.pos_topo = np.array(self.pos_topo)
        self.subs_ids = np.array(self.subs_ids)
        self.n = self.subs_ids.shape[0]

        if "max_sub_changed" in kwargs:
            self.max_sub_changed = int(kwargs["max_sub_changed"])

    def _get_id_from_obj(self, id_,
                         c_pos, g_pos, lor_pos, lex_pos,
                         next_load, next_gen, next_lor, next_lex):
        if id_ in c_pos:
            type_ = "load"
            id_obj_ = next_load
        elif id_ in g_pos:
            type_ = "gen"
            id_obj_ = next_gen
        elif id_ in lor_pos:
            type_ = "line_or"
            id_obj_ = next_lor
        elif id_ in lex_pos:
            type_ = "line_ex"
            id_obj_ = next_lex
        else:
            raise RuntimeError("Invalid grid")
        return type_, id_obj_

    def _get_pos_topo(self, type_, id_obj):
        if type_ == "load":
            res = self.load_pos_topo_vect[id_obj]
        elif type_ == "gen":
            res = self.gen_pos_topo_vect[id_obj]
        elif type_ == "line_or":
            res = self.line_or_pos_topo_vect[id_obj]
        elif type_ == "line_ex":
            res = self.line_ex_pos_topo_vect[id_obj]
        else:
            raise RuntimeError("Invalid grid")
        return res

    def convert_obs(self, obs):
        """
        This function is used to convert an observation into something that is easier to manipulate.

        **VERY IMPORTANT**: for this converter to work, it needs to remember the previous state of the grid, so you absolutely need to call its method :func:`ConnectivityConverter.convert_obs` at each observation.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The input observation.

        Returns
        -------

        transformed_obs: ``object``
            An different representation of the input observation, typically represented as a 1d vector that can be processed by a neural networks.

        """
        self.last_obs = obs
        return obs

    def convert_act(self, encoded_act):
        """
        For this converter, encoded_act is a vector, with the same size as there are possible ways to reconfigure
        the grid. And it find a consistent state that does not break too much the connectivity asked.

        NOTE: there might be better ways to do it...
        Parameters
        ----------
        encoded_act: ``numpy.ndarray``
            This action should have the same size as the number of pairs of element connectable.

        Returns
        -------
        act: :class:`grid2op.Action.BaseAction`
            The action that is usable by grid2op (after conversion) [the action space must be compatible with
            the "set_bus" key word]

        """
        argsort = np.argsort(np.minimum(encoded_act, 1-encoded_act))
        topo_vect = np.zeros(self.dim_topo, dtype=dt_int)
        subs_added = np.full(self.n_sub, fill_value=False)
        sub_changed = 0
        for el in argsort:
            my_sub = self.subs_ids[el]
            if not subs_added[my_sub]:
                subs_added[my_sub] = True
                topo_vect[self.pos_topo[el, 0]] = 1   # todo with self.last_obs !
                sub_changed += 1
                if sub_changed >= self.max_sub_changed:
                    break

        for el in argsort:
            bus_1_id = self.pos_topo[el, 0]
            bus_2_id = self.pos_topo[el, 1]
            need_1 = topo_vect[bus_1_id] <= 0
            need_2 = topo_vect[bus_2_id] <= 0
            val = encoded_act[el]
            if need_2 and not need_1:
                if val > 0.5:
                    # they are on same bus
                    topo_vect[bus_2_id] = topo_vect[bus_1_id]
                else:
                    # they are on different bus
                    topo_vect[bus_2_id] = 1 - topo_vect[bus_1_id] + 2
            elif need_1 and not need_2:
                if val > 0.5:
                    # they are on same bus
                    topo_vect[bus_1_id] = topo_vect[bus_2_id]
                else:
                    # they are on different bus
                    topo_vect[bus_1_id] = 1 - topo_vect[bus_2_id] + 2

        act = super().__call__({"set_bus": topo_vect})
        self.last_disagreement = self._compute_disagreement(encoded_act, topo_vect)
        return act

    def _compute_disagreement(self, encoded_act, topo_vect):
        """
         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        computes the disagreement between the encoded act and the proposed topo_vect

        **NB** if encoded act is random uniform, and topo_vect is full of 1, then disagreement is, on average 0.5.
        Lower disagreement is better.
        """
        bus_1 = topo_vect[self.pos_topo[:, 0]]
        bus_2 = topo_vect[self.pos_topo[:, 1]]
        together = 1. - encoded_act[bus_1 == bus_2]
        split = encoded_act[bus_1 != bus_2]
        raw_disag = together.sum() + split.sum()
        scaled_disag = raw_disag / self.n  # to have something between 0 and 1
        return scaled_disag

    def sample(self):
        coded_act = self.space_prng.rand(self.n)
        return self.convert_act(coded_act)
