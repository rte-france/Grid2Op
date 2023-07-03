# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from grid2op.Action import ActionSpace


class Converter(ActionSpace):
    """
    This Base class should be use to implement any converter. If for some reasons
    """

    def __init__(self, action_space):
        ActionSpace.__init__(
            self, action_space, action_space.legal_action, action_space.subtype
        )
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

    def get_gym_dict(self, cls_gym):
        """
        To convert this space into a open ai gym space. This function returns a dictionnary used
        to initialize such a converter.

        It should not be used directly. Prefer to use the :class:`grid2op.Converter.GymConverter`

        cls_gym represents either :class:`grid2op.gym_compat.LegacyGymActionSpace` or
        :class:`grid2op.gym_compat.GymnasiumActionSpace`
        """
        raise NotImplementedError(
            'Impossible to convert the converter "{}" automatically '
            "into a gym space (or gym is not installed on your machine)."
            "".format(self)
        )

    def convert_action_from_gym(self, gymlike_action):
        """
        Convert the action (represented as a gym object, in fact an ordered dict) as an action
        compatible with this converter.

        This is not compatible with all converters and you need to install gym for it to work.

        Parameters
        ----------
        gymlike_action:
            the action to be converted to an action compatible with the action space representation

        Returns
        -------
        res:
            The action converted to be understandable by this converter.

        Examples
        ---------
        Here is an example on how to use this feature with the :class:`grid2op.Converter.IdToAct`
        converter (imports are not shown here).

        .. code-block:: python

            # create the environment
            env = grid2op.make()

            # create the converter
            converter = IdToAct(env.action_space)

            # create the gym action space
            gym_action_space = GymObservationSpace(action_space=converter)

            gym_action = gym_action_space.sample()
            converter_action = converter.convert_action_from_gym(gym_action)  # this represents the same action
            grid2op_action = converter.convert_act(converter_action)

        """
        raise NotImplementedError(
            "Impossible to convert the gym-like action automatically "
            'into the converter representation for "{}" '
            "".format(self)
        )

    def convert_action_to_gym(self, action):
        """
        Convert the action (compatible with this converter) into a "gym action" (ie an OrderedDict)

        This is not compatible with all converters and you need to install gym for it to work.

        Parameters
        ----------
        action:
            the action to be converted to an action compatible with the action space representation

        Returns
        -------
        res:
            The action converted to a "gym" model (can be used by a machine learning model)

        Examples
        ---------
        Here is an example on how to use this feature with the :class:`grid2op.Converter.IdToAct`
        converter (imports are not shown here).

        .. code-block:: python

            # create the environment
            env = grid2op.make()

            # create the converter
            converter = IdToAct(env.action_space)

            # create the gym action space
            gym_action_space = GymObservationSpace(action_space=converter)

            converter_action = converter.sample()
            gym_action = converter.to_gym(converter_action)  # this represents the same action

        """
        raise NotImplementedError(
            "Impossible to convert the gym-like action automatically "
            'into the converter representation for "{}" '
            "".format(self)
        )
