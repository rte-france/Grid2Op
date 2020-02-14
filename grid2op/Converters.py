"""
In this module of grid2op, the "converters" are defined.

A converter is a specific class of :class:`grid2op.Action.HelperAction` (ie of Action Space) that allows the agent to
manipulate this action to have a different representation of it.

For example, suppose we are dealing with TopologyAction (only manipulating the graph of the powergrid). This is a
discrete "action space". Often, it's custom to deal with such action space by enumerating all actions, and then assign
to all valid actions a unique ID.

This can be done easily with the :class:`IdToAct` class.

More concretely, the diagram of an agent is:

i) receive an observation (in a form of an object of class :class:`grid2op.Observation.Observation`)
ii) implement the :func:`grid2op.Agent.Agent.act` taking as input an :class:`grid2op.Observation.Observation` and
    returning an :class:`grid2op.Action.Action`
iii) this :class:`grid2op.Action.Action` is then digested by the environment

Introducing some converters lead to the following:

i) receive an observation (:class:`grid2op.Observation.Observation`)
ii) the transformer automatically (using :func:`Converter.convert_obs`) to a `transformed observation`
iii) implement the function :func:`grid2op.Agent.AgentWithConverter.my_act` that takes as input
     a `transformed observation` and returns an `encoded action`
iv) the transformer automatically transforms back the `encoded action` into a proper :class:`grid2op.Action.Action`
v) this :class:`grid2op.Action.Action` is then digested by the environment

This simple mechanism allows people to focus on iii) above (typically implemented with artificial neural networks)
without having to worry each time about the complex representations of actions and observations.

More details and a concrete example is given in the documentation of the class
:class:`grid2op.Agent.AgentWithConverter`.

Some examples of converters are given in :class:`IdToAct` and :class:`ToVect`.
"""

import numpy as np
import itertools

try:
    from .Action import HelperAction
    from .Exceptions import Grid2OpException
except (ModuleNotFoundError, ImportError):
    from Action import HelperAction
    from Exceptions import Grid2OpException

import pdb

# TODO more exhaustive documentation and tests.


class Converter(HelperAction):
    """
    This Base class should be use to implement any converter. If for some reasons
    """
    def __init__(self, action_space):
        HelperAction.__init__(self, action_space, action_space.legal_action, action_space.subtype)
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

    Examples of non unary actions include:
    - disconnection / reconnection of 2 or more powerlines
    - change of the configuration of 2 or more substations
    - disconnection / reconnection of a single powerline and change of the configration of a single substation

    **NB** All the actions created automatically are unary. For the L2RPN 2019, agent could be allowed to act with non
    unary actions, for example by disconnecting a powerline and reconfiguring a substation. This class would not
    allow to do such action at one time step.

    **NB** The actions that are initialized by default uses the "set" way and not the "change" way (see the description
    of :class:`grid2op.Action.Action` for more information).

    For each powerline, 5 different actions will be computed:

    - disconnect it
    - reconnect it and connect it to bus 1 on "origin" end ann bus 1 on "extremity" end
    - reconnect it and connect it to bus 1 on "origin" end ann bus 2 on "extremity" end
    - reconnect it and connect it to bus 2 on "origin" end ann bus 1 on "extremity" end
    - reconnect it and connect it to bus 2 on "origin" end ann bus 2 on "extremity" end

    Actions corresponding to all topologies are also used by default. See
    :func:`grid2op.Action.HelperAction.get_all_unitary_topologies_set` for more information.


    In this converter:

    - `encoded_act` are positive integer, representing the index of the actions.
    - `transformed_obs` are regular observations.

    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.all_actions = []
        self.all_actions.append(super().__call__())  # add the do nothing topology
        self.n = 1

    def init_converter(self, all_actions=None, **kwargs):
        """
        This function is used to initialized the converter. When the converter is created, this method should be called
        otherwise the converter might be in an unstable state.

        Parameters
        ----------
        all_actions: ``list``
            The (ordered) list of all actions that the agent will be able to perform. If given a number ``i`` the
            converter will return action ``all_actions[i]``. In the "pacman" game, this vector could be
            ["up", "down", "left", "right"], in this case "up" would be encode by 0, "down" by 1, "left" by 2 and
            "right" by 3. If nothing is provided, the converter will output all the unary actions possible for
            the environment. Be careful, computing all these actions might take some time.

        kwargs:
            other keyword arguments

        """
        if all_actions is None:
            self.all_actions = []
            if "_set_line_status" in self._template_act.attr_list_vect:
                # powerline switch: disconnection
                for i in range(self.n_line):
                    self.all_actions.append(self.disconnect_powerline(line_id=i))

                # powerline switch: reconnection
                for bus_or in [1, 2]:
                    for bus_ex in [1, 2]:
                        for i in range(self.n_line):
                            tmp_act = self.reconnect_powerline(line_id=i, bus_ex=bus_ex, bus_or=bus_or)
                            self.all_actions.append(tmp_act)

            if "_set_topo_vect" in self._template_act.attr_list_vect:
                # topologies using the 'set' method
                self.all_actions += self.get_all_unitary_topologies_set(self)
            elif "_change_bus_vect" in self._template_act.attr_list_vect:
                # topologies 'change'
                self.all_actions += self.get_all_unitary_topologies_change(self)

        else:
            self.all_actions = all_actions
        self.n = len(self.all_actions)

    def sample(self):
        """
        Having define a complete set of observation an agent can do, sampling from it is now made easy.

        One action amoung the n possible actions is used at random.

        Returns
        -------
        res: ``int``
            An id of an action.

        """
        idx = self.space_prng.randint(0, self.n)
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
        This converter will match the observation to a vector, using the :func:`grid2op.Observation.Observation.to_vect`
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
            The corresponding action transformed with the :func:`grid2op.Action.Action.from_vect`.

        """
        res = self.__call__({})
        res.from_vect(encoded_act)
        return res