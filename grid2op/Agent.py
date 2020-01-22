"""
In this RL framework, an :class:`Agent` is an entity that acts on the :class:`Environment`. Agent can alternatively
be named "bot" or "controller" in other literature.

This module presents a few possible :class:`Agent` that can serve either as baseline, or as example on how to
implement such agents.

To perform their actions, agent receive two main signals from the :class:`grid2op.Environment`:

  - the :class:`grid2op.Reward` that states how good the previous has been
  - the :class:`grid2op.Observation` that is a (partial) view on the state of the Environment.

Both these signals can be use to determine what is the best action to perform on the grid. This is actually the main
objective of an :class:`Agent`, and this is done in the :func:`Agent.act` method.

"""

from abc import ABC, abstractmethod
import numpy as np
import itertools
import pdb

try:
    from .Converters import Converter, IdToAct, ToVect
    from .Exceptions import Grid2OpException
except (ModuleNotFoundError, ImportError):
    from ActionSpaceConverter import Converter, IdToAct, ToVect
    from Exceptions import Grid2OpException


class Agent(ABC):
    """
    This class represents the base class of an Agent. All bot / controller / agent used in the Grid2Op simulator
    should derived from this class.

    To work properly, it is advise to create Agent after the :class:`grid2op.Environment` has been created and reuse
    the :attr:`grid2op.Environment.Environment.action_space` to build the Agent.

    Attributes
    -----------
    action_space: :class:`grid2op.Action.HelperAction`
        It represent the action space ie a tool that can serve to create valid action. Note that a valid action can
        be illegal or ambiguous, and so lead to a "game over" or to a error. But at least it will have a proper size.

    """
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def act(self, observation, reward, done=False):
        """
        This is the main method of an Agent. Given the current observation and the current reward (ie the reward that
        the environment send to the agent after the previous action has been implemented).

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controler / agent.

        """
        pass


class DoNothingAgent(Agent):
    """
    This is the most basic Agent. It is purely passive, and does absolutely nothing.
    """
    def __init__(self, action_space):
        Agent.__init__(self, action_space)

    def act(self, observation, reward, done=False):
        """
        As better explained in the document of :func:`grid2op.Action.update` or
        :func:`grid2op.Action.HelperAction.__call__`.

        The preferred way to make an object of type action is to call :func:`grid2op.Action.HelperAction.__call__` with
        the
        dictionnary representing the action. In this case, the action is "do nothing" and it is represented by the
        empty dictionnary.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        res = self.action_space({})
        return res


class OneChangeThenNothing(Agent):
    """
    This is a specific kind of Agent. It does an Action (possibly non empty) at the first time step and then does
    nothing.

    This class is an abstract class and cannot be instanciated (ie no object of this class can be created). It must
    be overridden and the method :func:`OneChangeThenNothing._get_dict_act` be defined. Basically, it must know
    what action to do.

    """
    def __init__(self, action_space, action_space_converter=None):
        Agent.__init__(self, action_space)
        self.has_changed = False

    def act(self, observation, reward, done=False):
        if self.has_changed:
            res = self.action_space({})
            self.has_changed = True
        else:
            res = self.action_space(self._get_dict_act())
        return res

    @abstractmethod
    def _get_dict_act(self):
        """
        Function that need to be overridden to indicate which action to perfom.

        Returns
        -------
        res: ``dict``
            A dictionnary that can be converted into a valid :class:`grid2op.Action.Action`. See the help of
            :func:`grid2op.Action.HelperAction.__call__` for more information.
        """
        pass


class GreedyAgent(Agent):
    """
    This is a class of "Greedy Agent". Greedy agents are all executing the same kind of algorithm to take action:

      1. They :func:`grid2op.Observation.simulate` all actions in a given set
      2. They take the action that maximise the simulated reward among all these actions

    To make the creation of such Agent, we created this abstract class (object of this class cannot be created). Two
    examples of such greedy agents are provided with :class:`PowerLineSwitch` and :class:`TopologyGreedy`.
    """
    def __init__(self, action_space, action_space_converter=None):
        Agent.__init__(self, action_space)
        self.tested_action = None

    def act(self, observation, reward, done=False):
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        self.tested_action = self._get_tested_action(observation)
        if len(self.tested_action) > 1:
            all_rewards = np.full(shape=len(self.tested_action), fill_value=np.NaN, dtype=np.float)
            for i, action in enumerate(self.tested_action):
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                all_rewards[i] = simul_reward

            reward_idx = np.argmax(all_rewards)  # rewards.index(max(rewards))
            best_action = self.tested_action[reward_idx]
        else:
            best_action = self.tested_action[0]
        return best_action

    @abstractmethod
    def _get_tested_action(self, observation):
        """
        Returns the list of all the candidate actions.

        From this list, the one that achieve the best "simulated reward" is used.


        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        Returns
        -------
        res: ``list``
            A list of all candidate :class:`grid2op.Action.Action`
        """
        pass


class PowerLineSwitch(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to disconnect powerlines.

    It will choose among:

      - doing nothing
      - disconnecting one powerline

    which action that will maximize the reward. All powerlines are tested.

    """

    def __init__(self, action_space):
        GreedyAgent.__init__(self, action_space)

    def _get_tested_action(self, observation):
        res = [self.action_space({})]  # add the do nothing
        for i in range(self.action_space.n_line):
            tmp = np.full(self.action_space.n_line, fill_value=False, dtype=np.bool)
            tmp[i] = True
            action = self.action_space({"change_line_status": tmp})
            if not observation.line_status[i]:
                # so the action consisted in reconnecting the powerline
                # i need to say on which bus (always on bus 1 for this type of agent)
                action = action.update({"set_bus": {"lines_or_id": [(i, 1)], "lines_ex_id": [(i, 1)]}})
            res.append(action)
        return res


class TopologyGreedy(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to reconfigure the substations connectivity.

    It will choose among:

      - doing nothing
      - changing the topology of one substation.

    """
    def __init__(self, action_space, action_space_converter=None):
        GreedyAgent.__init__(self, action_space, action_space_converter=action_space_converter)
        self.li_actions = None

    def _get_tested_action(self, observation):
        if self.li_actions is None:
            res = [self.action_space({})]  # add the do nothing
            res += self.action_space.get_all_unitary_topologies_change(self.action_space)
            self.li_actions = res
        return self.li_actions


class AgentWithConverter(Agent):
    def __init__(self, action_space, action_space_converter=None, **kwargs_converter):
        self.action_space_converter = action_space_converter
        self.init_action_space = action_space

        if action_space_converter is None:
            Agent.__init__(self, action_space)
        else:
            if isinstance(action_space_converter, type):
                if issubclass(action_space_converter, Converter):
                    Agent.__init__(self, action_space_converter(action_space))
                else:
                    raise Grid2OpException("Impossible to make an Agent with a converter of type {}. "
                                           "Please use a converter deriving from grid2op.ActionSpaceConverter.Converter."
                                           "".format(action_space_converter))
            elif isinstance(action_space_converter, Converter):
                if isinstance(action_space_converter.template_act, self.init_action_space.template_act):
                    Agent.__init__(self, action_space_converter)
                else:
                    raise Grid2OpException("Impossible to make an Agent with the provided converter of type {}. "
                                           "It doesn't use the same type of action as the Agent's action space."
                                           "".format(action_space_converter))
            else:
                raise Grid2OpException("You try to initialize and Agent with an invalid converter \"{}\". It must"
                                       "either be a type deriving from \"Converter\", or an instance of a class"
                                       "deriving from it."
                                       "".format(action_space_converter))
            self.action_space.init_actions(**kwargs_converter)

    def convert_obs(self, observation):
        return self.action_space.convert_obs(observation)

    def convert_act(self, act):
        return self.action_space.convert_act(act)

    def act(self, observation, reward, done=False):
        convert_obs = self.convert_obs(observation)
        act = self.my_act(convert_obs, reward, done)
        return self.convert_act(act)

    @abstractmethod
    def my_act(self, transformed_observation, reward, done=False):
        transformed_action = None
        return transformed_action


class RandomAgent(AgentWithConverter):
    def __init__(self, action_space, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)

    def my_act(self, transformed_observation, reward, done=False):
        return self.action_space.sample()


class MLAgent(AgentWithConverter):
    def __init__(self, action_space, action_space_converter=ToVect, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)
        self.do_nothing_vect = action_space({}).to_vect()

    def my_act(self, transformed_observation, reward, done=False):
        return self.do_nothing_vect

    def convert_from_vect(self, act):
        """
        Helper to convert an action, represented as a numpy array as an :class:`grid2op.Action` instance.

        Parameters
        ----------
        act: ``numppy.ndarray``
            An action cast as an :class:`grid2op.Action.Action` instance.

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The `act` parameters converted into a proper :class:`grid2op.Action.Action` object.
        """
        res = self.action_space({})
        res.from_vect(act)
        return res

# class MLAgent(Agent):
#     """
#     This Agent class has the particularity to handle only vector representation of :class:`grid2op.Action.Action` and
#     :class:`grid2op.Observation.Observation`. It is particularly suited for building Machine Learning Agents.
#
#     Attributes
#     ----------
#     do_nothing_vect: ``numpy.ndarray``, dtype:np.float
#         The representation of the "do nothing" Action as a numpy vector.
#
#     """
#     def __init__(self, action_space):
#         Agent.__init__(self, action_space)
#         self.do_nothing_vect = action_space({}).to_vect()
#
#     def convert_from_vect(self, act):
#         """
#         Helper to convert an action, represented as a numpy array as an :class:`grid2op.Action` instance.
#
#         Parameters
#         ----------
#         act: ``numppy.ndarray``
#             An action cast as an :class:`grid2op.Action.Action` instance.
#
#         Returns
#         -------
#         res: :class:`grid2op.Action.Action`
#             The `act` parameters converted into a proper :class:`grid2op.Action.Action` object.
#         """
#         res = self.action_space({})
#         res.from_vect(act)
#         return res
#
#     def act(self, observation, reward, done=False):
#         """
#         Overloading of the method `act` to deal with vectors representation of :class:`grid2op.Observation` and
#         :class:`grid2op.Action` rather than with class.
#
#         Parameters
#         ----------
#         observation: :class:`grid2op.Observation.Observation`
#             The current observation of the :class:`grid2op.Environment`
#
#         reward: ``float``
#             The current reward. This is the reward obtained by the previous action
#
#         done: ``bool``
#             Whether the episode has ended or not. Used to maintain gym compatibility
#
#         Returns
#         -------
#         res: :class:`grid2op.Action.Action`
#             The action chosen by the bot / controler / agent.
#
#         """
#         obs = observation.to_vect()
#         act = self._ml_act(obs, reward, done=done)
#         return self.convert_from_vect(act)
#
#     def _ml_act(self, observation, reward, done=False):
#         """
#         The method to modify that is able to handle numpy array representation of both action and observation.
#         It should be overidden for ML agents using the :class:`MLAgent`.
#
#         Parameters
#         ----------
#         observation: ``numppy.ndarray``
#             The current observation of the game, represented as a numpy array vector.
#
#         reward: ``float``
#             The current reward. This is the reward obtained by the previous action
#
#         done: ``bool``
#             Whether the episode has ended or not. Used to maintain gym compatibility
#
#         Returns
#         -------
#         res: ``numppy.ndarray``
#             The action taken at time t, represented as a numpy array vector.
#         """
#
#         return self.do_nothing_vect

