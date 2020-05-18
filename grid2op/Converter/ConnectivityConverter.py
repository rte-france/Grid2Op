# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np

from grid2op.Action import BaseAction
from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_float, dt_int
import pdb


class ConnectivityConverter(Converter):
    """
    In this converter, you have as many output as pairs of object that can be connected, and your model is asked
    to output 0 if he wants these elements disconnected and 1 if he wants them otherwise.

    This type of modeling is rather hard to "get working" the first time, especially because some "conflict" might
    appear. For example, consider three objects (line for example) on a given substation. You can chose to "connect
    A and B", connect "B and C" but "**not connect** A and C" in this case you need an algorithm to disambuate your
    action.

    **NB** compare to :class:`IdToAct` this converter allows for a smaller size. If you have N elements connected at
    a substation, you end up with `N*(N-1)/2` different action. Compare to IdToAct though, it is expected that your
    algorithm produces more than 1 output.

    **/!\ VERY IMPORTANT /!\** : for this converter to work, it needs to remember the previous state of the grid, so you
    absolutely need to call its method :func:`ConnectivityConverter.convert_obs` a each observation.
    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = ConnectivityConverter.init_grid(action_space)
        self.subs_ids = np.array([], dtype=dt_int)
        self.obj_type = []
        self.n = 1
        self.last_obs = None

    def init_converter(self, all_actions=None, **kwargs):
        # compute all pairs of elements that can be connected together
        for sub_id, nb_element in enumerate(self.sub_info):
            if nb_element <= 4:
                continue
            nb_pairs = int(nb_element * (nb_element - 1)/2)
            self.subs_ids = np.concatenate((self.subs_ids, sub_id * np.ones(nb_pairs, dtype=dt_int)))

            c_id = np.where(self.load_to_subid == sub_id)[0]
            g_id = np.where(self.gen_to_subid == sub_id)[0]
            lor_id = np.where(self.line_or_to_subid == sub_id)[0]
            lex_id = np.where(self.line_ex_to_subid == sub_id)[0]

            c_pos = self.load_to_sub_pos[self.load_to_subid == sub_id]
            g_pos = self.gen_to_sub_pos[self.gen_to_subid == sub_id]
            lor_pos = self.line_or_to_sub_pos[self.line_or_to_subid == sub_id]
            lex_pos = self.line_ex_to_sub_pos[self.line_ex_to_subid == sub_id]

            my_types = []
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
            for id_i in range(nb_element):
                id_i_ = my_types[id_i]
                for id_j in range(id_i+1, nb_element):
                    id_j_ = my_types[id_j]
                    self.obj_type.append((sub_id, id_i_, id_j_))
        self.n = self.subs_ids.shape[0]

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

    def convert_obs(self, obs):
        """
        This function is used to convert an observation into something that is easier to manipulate.

        **/!\ VERY IMPORTANT /!\** : for this converter to work, it needs to remember the previous state of the grid,
        so you
        absolutely need to call its method :func:`ConnectivityConverter.convert_obs` a each observation.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The input observation.

        Returns
        -------

        transformed_obs: ``object``
            An different representation of the input observation, typically represented as a 1d vector that can be
            processed by a neural networks.

        """
        self.last_obs = obs
        return obs

    def convert_act(self, encoded_act):
        """
        For this converter, encoded_act is a vector, with the same size as there are possible ways to reconfigure
        the grid. And it find a consistent state that does not break too much the connectivity asked.

        Parameters
        ----------
        encoded_act

        Returns
        -------

        """
        return super().__call__()

