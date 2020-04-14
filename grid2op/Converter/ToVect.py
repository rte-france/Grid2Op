# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Converter.Converters import Converter

import pdb


class ToVect(Converter):
    """
    This converters allows to manipulate the vector representation of the actions and observations.

    In this converter:

    - `encoded_act` are numpy ndarray
    - `transformed_obs` are numpy ndarray

    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.do_nothing_vect = action_space({}).to_vect()

    def convert_obs(self, obs):
        """
        This converter will match the observation to a vector, using the
        :func:`grid2op.Observation.BaseObservation.to_vect`
        function.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The observation, that will be processed into a numpy ndarray vector.

        Returns
        -------
        transformed_obs: ``numpy.ndarray``
            The vector representation of the action.

        """
        return obs.to_vect()

    def convert_act(self, encoded_act):
        """
        In this converter `encoded_act` is a numpy ndarray. This function transforms it back to a valid action.

        Parameters
        ----------
        encoded_act: ``numpy.ndarray``
            The action, representated as a vector

        Returns
        -------
        regular_act: :class:`grid2op.Action.Action`
            The corresponding action transformed with the :func:`grid2op.Action.BaseAction.from_vect`.

        """
        res = self.__call__({})
        res.from_vect(encoded_act)
        return res
