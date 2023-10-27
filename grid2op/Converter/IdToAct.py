# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np
from collections import OrderedDict

from grid2op.Action import BaseAction
from grid2op.Converter.Converters import Converter
from grid2op.Exceptions import Grid2OpException
from grid2op.dtypes import dt_float, dt_int, int_types


class IdToAct(Converter):
    """
    This type of converter allows to represent action with unique id. Instead of manipulating complex objects, it allows
    to manipulate only positive integer.

    The list of all actions can either be the list of all possible unary actions (see below for a complete
    description) or by a given pre computed list.

    A "unary action" is an action that consists only in acting on one "concept" it includes:

    - disconnecting a single powerline
    - reconnecting a single powerline and connect it to bus xxx on its origin end and yyy on its extremity end
    - changing the topology of a single substation
    - performing redispatching on a single generator
    - performing curtailment on a single generator
    - performing action on a single storage unit

    Examples of non unary actions include:
    - disconnection / reconnection of 2 or more powerlines
    - change of the configuration of 2 or more substations
    - disconnection / reconnection of a single powerline and change of the configration of a single substation

    **NB** All the actions created automatically are unary. For the L2RPN 2019, agent could be allowed to act with non
    unary actions, for example by disconnecting a powerline and reconfiguring a substation. This class would not
    allow to do such action at one time step.

    **NB** The actions that are initialized by default uses the "set" way and not the "change" way (see the description
    of :class:`grid2op.BaseAction.BaseAction` for more information).

    For each powerline, 5 different actions will be computed:

    - disconnect it
    - reconnect it and connect it to bus 1 on "origin" end ann bus 1 on "extremity" end
    - reconnect it and connect it to bus 1 on "origin" end ann bus 2 on "extremity" end
    - reconnect it and connect it to bus 2 on "origin" end ann bus 1 on "extremity" end
    - reconnect it and connect it to bus 2 on "origin" end ann bus 2 on "extremity" end

    Actions corresponding to all topologies are also used by default. See
    :func:`grid2op.BaseAction.ActionSpace.get_all_unitary_topologies_set` for more information.

    In this converter:

    - `encoded_act` are positive integer, representing the index of the actions.
    - `transformed_obs` are regular observations.

    **NB** The number of actions in this converter can be especially big. For example, if a substation counts N elements
    there are roughly 2^(N-1) possible actions in this substation. This means if there are a single substation with
    more than N = 15 or 16 elements, the amount of actions (for this substation alone) will be higher than 16.000
    which makes it rather difficult to handle for most machine learning algorithm. Be carefull with that !

    """

    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = IdToAct.init_grid(action_space)
        self.all_actions = []
        # add the do nothing topology
        self.all_actions.append(super().__call__())
        self.n = 1
        self._init_size = action_space.size()
        self.kwargs_init = {}

    def init_converter(self, all_actions=None, **kwargs):
        """
        This function is used to initialized the converter. When the converter is created, this method should be called
        otherwise the converter might be in an unstable state.

        Parameters
        ----------
        all_actions: ``None``, ``list``, ``str``, ``np.ndarray``
            See the example section for more informations.

            if `all_actions` is:

                - ``None``: the action space will be built from scratch using the provided key word arguments.
                - a ``list``: The (ordered) list of all actions that the agent will be able to perform.
                  If given a number ``i`` the
                  converter will return action ``all_actions[i]``. In the "pacman" game, this vector could be
                  ["up", "down", "left", "right"], in this case "up" would be encode by 0, "down" by 1, "left" by 2 and
                  "right" by 3. If nothing is provided, the converter will output all the unary actions possible for
                  the environment. Be careful, computing all these actions might take some time.
                - a ``str`` this will be considered as a path where a previous converter has been saved. You need to
                  provide the full path, including the filename and its extension. It gives something like:
                  "/path/where/it/is/saved/action_space_vect.npy"

        kwargs:
            other keyword arguments (all considered to be ``True`` by default) that can be:

            set_line_status: ``bool``
                Whether you want to include the set line status in your action
                (in case the original action space allows it)

            change_line_status: ``bool``
                Whether you want to include the "change line status" in your action space
                (in case the original action space allows it)

            change_line_status: ``bool``
                Whether you want to include the "change line status" in your action space
                (in case the original action space allows it)

            set_topo_vect: ``bool``
                Whether you want to include the "set_bus" in your action space
                (in case the original action space allows it)

            change_bus_vect: ``bool``
                Whether you want to include the "change_bus" in your action space
                (in case the original action space allows it)

            redispatch: ``bool``
                Whether you want to include the "redispatch" in your action space
                (in case the original action space allows it)

            curtail: ``bool``
                Whether you want to include the "curtailment" in your action space
                (in case the original action space allows it)

            storage: ``bool``
                Whether you want to include the "storage unit" in your action space
                (in case the original action space allows it)

        Examples
        --------
        Here is an example of a code that will: make a converter by selecting some action. Save it, and then restore
        its original state to be used elsewhere.

        .. code-block:: python

            import grid2op
            from grid2op.Converter import IdToAct
            env = grid2op.make("l2rpn_case14_sandbox")
            converter = IdToAct(env.action_space)

            # the path were will save it
            path_ = "/path/where/it/is/saved/"
            name_file = "tmp_convert.npy"

            # init the converter, the first time, here by passing some key word arguments, to not consider
            # redispatching for example
            converter.init_converter(redispatch=False)
            converter.save(path_, name_file)

            # i just do an action, for example the number 27... whatever it does does not matter here
            act = converter.convert_act(27)

            converter2 = IdToAct(self.env.action_space)
            converter2.init_converter(all_actions=os.path.join(path_, name_file))
            act2 = converter2.convert_act(27)

            assert act ==  act2  # this is ``True`` the converter has properly been saved.


        """
        self.kwargs_init = kwargs
        if all_actions is None:
            self.all_actions = []
            # add the do nothing action, always
            self.all_actions.append(super().__call__())
            if "_set_line_status" in self._template_act.attr_list_vect:
                # lines 'set'
                include_ = True
                if "set_line_status" in kwargs:
                    include_ = kwargs["set_line_status"]
                if include_:
                    self.all_actions += self.get_all_unitary_line_set(self)

            if "_switch_line_status" in self._template_act.attr_list_vect:
                # lines 'change'
                include_ = True
                if "change_line_status" in kwargs:
                    include_ = kwargs["change_line_status"]
                if include_:
                    self.all_actions += self.get_all_unitary_line_change(self)

            if "_set_topo_vect" in self._template_act.attr_list_vect:
                # topologies 'set'
                include_ = True
                if "set_topo_vect" in kwargs:
                    include_ = kwargs["set_topo_vect"]
                if include_:
                    self.all_actions += self.get_all_unitary_topologies_set(self)

            if "_change_bus_vect" in self._template_act.attr_list_vect:
                # topologies 'change'
                include_ = True
                if "change_bus_vect" in kwargs:
                    include_ = kwargs["change_bus_vect"]
                if include_:
                    self.all_actions += self.get_all_unitary_topologies_change(self)

            if "_redispatch" in self._template_act.attr_list_vect:
                # redispatch (transformed to discrete variables)
                include_ = True
                if "redispatch" in kwargs:
                    include_ = kwargs["redispatch"]
                if include_:
                    self.all_actions += self.get_all_unitary_redispatch(self)

            if "_curtail" in self._template_act.attr_list_vect:
                # redispatch (transformed to discrete variables)
                include_ = True
                if "curtail" in kwargs:
                    include_ = kwargs["curtail"]
                if include_:
                    self.all_actions += self.get_all_unitary_curtail(self)

            if "_storage_power" in self._template_act.attr_list_vect:
                # redispatch (transformed to discrete variables)
                include_ = True
                if "storage" in kwargs:
                    include_ = kwargs["storage"]
                if include_:
                    self.all_actions += self.get_all_unitary_storage(self)

        elif isinstance(all_actions, str):
            # load the path from the path provided
            if not os.path.exists(all_actions):
                raise FileNotFoundError(
                    'No file located at "{}" where the actions should have been stored.'
                    "".format(all_actions)
                )
            try:
                all_act = np.load(all_actions)
            except Exception as e:
                raise RuntimeError(
                    'Impossible to load the data located at "{}" with error\n{}.'
                    "".format(all_actions, e)
                )
            try:
                self.all_actions = np.array([self.__call__() for _ in all_act])
                for i, el in enumerate(all_act):
                    self.all_actions[i].from_vect(el)
            except Exception as e:
                raise RuntimeError(
                    'Impossible to convert the data located at "{}" into valid grid2op action. '
                    "The error was:\n{}".format(all_actions, e)
                )
        elif isinstance(all_actions, (list, np.ndarray)):
            # assign the action to my actions
            possible_act = all_actions[0]
            if isinstance(possible_act, BaseAction):
                # list of grid2op action
                self.all_actions = np.array(all_actions)
            elif isinstance(possible_act, dict):
                # list of dictionnary (obtained with `act.as_serializable_dict()`)
                self.all_actions = np.array([self.__call__(el) for el in all_actions])
            else:
                # should be an array !
                try:
                    self.all_actions = np.array([self.__call__() for _ in all_actions])
                    for i, el in enumerate(all_actions):
                        self.all_actions[i].from_vect(el)
                except Exception as exc_:
                    raise Grid2OpException(
                        'Impossible to convert the data provided in "all_actions" into valid '
                        "grid2op action. The error was:\n{}".format(e)
                    ) from exc_
        else:
            raise RuntimeError("Impossible to load the action provided.")
        self.n = len(self.all_actions)

    def filter_action(self, filtering_fun):
        """
        This function allows you to "easily" filter generated actions.

        **NB** the action space will change after a call to this function, especially its size. It is NOT recommended
        to apply it once training has started.

        Parameters
        ----------
        filtering_fun: ``function``
            This takes an action as input and should retrieve ``True`` meaning "this action will be kept" or
            ``False`` meaning "this action will be dropped.

        """
        self.all_actions = np.array(
            [el for el in self.all_actions if filtering_fun(el)]
        )
        self.n = len(self.all_actions)

    def save(self, path, name="action_space_vect.npy"):
        """
        Save the action space as a numpy array that can be reloaded afterwards with the :func:`IdToAct.init_converter`
        function by setting argument `all_actions` to `os.path.join(path, name)`

        The resulting object will be a numpy array of float. Each row of this array will be an action of the
        action space.

        Parameters
        ----------
        path: ``str``
            The path were to save the action space

        name: ``str``, optional
            The name of the numpy array stored on disk. By default its "action_space_vect.npy"

        Examples
        --------
        Here is an example of a code that will: make a converter by selecting some action. Save it, and then restore
        its original state to be used elsewhere.

        .. code-block:: python

            import grid2op
            from grid2op.Converter import IdToAct
            env = grid2op.make("l2rpn_case14_sandbox")
            converter = IdToAct(env.action_space)

            # the path were will save it
            path_ = "/path/where/it/is/saved/"
            name_file = "tmp_convert.npy"

            # init the converter, the first time, here by passing some key word arguments, to not consider
            # redispatching for example
            converter.init_converter(redispatch=False)
            converter.save(path_, name_file)

            # i just do an action, for example the number 27... whatever it does does not matter here
            act = converter.convert_act(27)

            converter2 = IdToAct(self.env.action_space)
            converter2.init_converter(all_actions=os.path.join(path_, name_file))
            act2 = converter2.convert_act(27)

            assert act ==  act2  # this is ``True`` the converter has properly been saved.

        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                'Impossible to save the action space as the directory "{}" does not exist.'
                "".format(path)
            )
        if not os.path.isdir(path):
            raise NotADirectoryError(
                'The path to save the action space provided "{}" is not a directory.'
                "".format(path)
            )
        saved_npy = (
            np.array([el.to_vect() for el in self.all_actions])
            .astype(dtype=dt_float)
            .reshape(self.n, -1)
        )
        np.save(file=os.path.join(path, name), arr=saved_npy)

    def sample(self):
        """
        Having define a complete set of observation an agent can do, sampling from it is now made easy.

        One action amoung the n possible actions is used at random.

        Returns
        -------
        res: ``int``
            An id of an action.

        """
        idx = self.space_prng.randint(0, self.n, dtype=dt_int)
        return idx

    def convert_act(self, encoded_act):
        """
        In this converter, we suppose that "encoded_act" is an id of an action stored in the
        :attr:`IdToAct.all_actions` list.

        Converting an id of an action (here called "act") into a valid action is then easy: we just need to take the
        "act"-th element of :attr:`IdToAct.all_actions`.

        Parameters
        ----------
        encoded_act: ``int``
            The id of the action

        Returns
        -------
        action: :class:`grid2op.Action.Action`
            The action corresponding to id "act"
        """

        return self.all_actions[encoded_act]

    def get_gym_dict(self, cls_gym):
        """
        Transform this converter into a dictionary that can be used to initialized a :class:`gym.spaces.Dict`.
        The converter is modeled as a "Discrete" gym space with as many elements as the number
        of different actions handled by this converter.

        This is available as the "action" keys of the spaces.Dict gym action space build from it.

        This function should not be used "as is", but rather through :class:`grid2op.Converter.GymConverter`

        cls_gym represents either :class:`grid2op.gym_compat.LegacyGymActionSpace` or
        :class:`grid2op.gym_compat.GymnasiumActionSpace`
        """
        res = {"action": cls_gym._DiscreteType(n=self.n)}
        return res

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
            env = grid2op.make("l2rpn_case14_sandbox")

            # create the converter
            converter = IdToAct(env.action_space)

            # create the gym action space
            gym_action_space = GymObservationSpace(action_space=converter)

            gym_action = gym_action_space.sample()
            converter_action = converter.from_gym(gym_action)  # this represents the same action
            grid2op_action = converter.convert_act(converter_action)  # this is a grid2op action

        """
        res = gymlike_action["action"]
        if not isinstance(res, int_types):
            raise RuntimeError("TODO")
        return int(res)

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
            env = grid2op.make("l2rpn_case14_sandbox")

            # create the converter
            converter = IdToAct(env.action_space)

            # create the gym action space
            gym_action_space = GymObservationSpace(action_space=converter)

            converter_action = converter.sample()
            gym_action = converter.to_gym(converter_action)  # this represents the same action

        """
        res = OrderedDict({"action": int(action)})
        return res
