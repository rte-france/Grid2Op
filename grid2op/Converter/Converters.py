# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from grid2op.Action import ActionSpace

import pdb

# TODO more exhaustive documentation and tests.


class Converter(ActionSpace):
    """
    This Base class should be use to implement any converter. If for some reasons
    """
    def __init__(self, action_space):
        ActionSpace.__init__(self, action_space, action_space.legal_action, action_space.subtype)
        self.space_prng = action_space.space_prng
        self.seed_used = action_space.seed_used

    def init_converter(self, **kwargs):
        pass

    def convert_obs(self, obs):
        """
        This function is used to convert an observation into something that is easier to manipulate.

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
        transformed_obs = obs
        return transformed_obs

    def convert_act(self, encoded_act):
        """
        This function will transform the action, encoded somehow (for example identified by an id, represented by
        an integer) to a valid actions that can be processed by the environment.

        Parameters
        ----------
        encoded_act: ``object``
            Representation of an action, as a vector or an integer etc.

        Returns
        -------
        regular_act: :class:`grid2op.Action.Action`
            The action corresponding to the `encoded_action` above converted into a format that can be processed
            by the environment.

        """
        regular_act = encoded_act
        return regular_act
