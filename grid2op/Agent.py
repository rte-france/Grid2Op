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
    from Converters import Converter, IdToAct, ToVect
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
            # print("reward_idx: {}".format(reward_idx))
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
    """
    Compared to a regular Agent, these types of Agents are able to deal with a different representation of
    :class:`grid2op.Action.Action` and :class:`grid2op.Observation.Observation`.

    As any other Agents, AgentWithConverter will implement the :func:`Agent.act` method. But for them, it's slightly
    different.

    They receive in this method an observation, as an object (ie an instance of :class:`grid2op.Observation`). This
    object can then be converted to any other object with the method :func:`AgentWithConverter.convert_obs`.

    Then, this `transformed_observation` is pass to the method :func:`AgentWithConverter.my_act` that is supposed
    to be defined for each agents. This function outputs an `encoded_act` which can be whatever you want to be.

    Finally, the `encoded_act` is decoded into a proper action, object of class :class:`grid2op.Action.Action`,
    thanks to the method :func:`AgentWithConverter.convert_act`.

    This allows, for example, to represent actions as integers to train more easily standard discrete control algorithm
    used to solve atari games for example.

    **NB** It is possible to define :func:`AgentWithConverter.convert_obs` and :func:`AgentWithConverter.convert_act`
     or to define a :class:`grid2op.Converters.Converter` and feed it to the `action_space_converter` parameters
     used to initialise the class. The second option is preferred, as the :attr:`AgentWithConverter.action_space`
     will then directly be this converter. Such an Agent will really behave as if the actions are encoded the way he
     wants.

    Examples
    --------
    For example, imagine an Agent uses a neural networks to take its decision.

    Suppose also that, after some
    features engineering, it's best for the neural network to use only the load active values
    (:attr:`grid2op.Observation.Observation.load_p`) and the sum of the
    relative flows (:attr:`grid2op.Observation.Observation.rho`) with the active flow
    (:attr:`grid2op.Observation.Observation.p_or`) [**NB** that agent would not make sense a priori, but who knows]

    Suppose that this neural network can be accessed with a class `AwesomeNN` (not available...) that can
    predict some actions. It can be loaded with the "load" method and make predictions with the
    "predict" method.

    For the sake of the examples, we will suppose that this agent only predicts powerline status (so 0 or 1) that
    are represented as vector. So we need to take extra care to convert this vector from a numpy array to a valid
    action.

    This is done below:


    .. code-block:: python

        import grid2op
        import AwesomeNN # this does not exists!
        # create a simple environment
        env = grid2op.make()

        # define the class above
        class AgentCustomObservation(AgentWithConverter):
            def __init__(self, action_space, path):
                AgentWithConverter.__init__(self, action_space)
                self.my_neural_network = AwesomeNN()
                self.my_neural_networl.load(path)

            def convert_obs(self, observation):
                # convert the observation
                return np.concatenate((observation.load_p, observation.rho + observation.p_or))

            def convert_act(self, encoded_act):
                # convert back the action, output from the NN "self.my_neural_network"
                # to a valid action
                act = self.action_space({"set_status": encoded_act})

            def my_act(self, transformed_observation, reward, done=False):
                act_predicted = self.my_neural_network(transformed_observation)
                return act_predicted


        # make the agent that behaves as expected.
        my_agent = AgentCustomObservation(action_space=env.action_space, path=".")

        # this agent is perfectly working :-) You can use it as any other agents.


    Attributes
    ----------
    action_space_converter: :class:`grid2op.Converters.Converter`
        The converter that is used to represents the Agent action space. Might be set to ``None`` if not initialized

    init_action_space: :class:`grid2op.Action.HelperAction`
        The initial action space. This corresponds to the action space of the :class:`grid2op.Environment.Environment`.

    action_space: :class:`grid2op.Converters.HelperAction`
        If a converter is used, then this action space represents is this converter. The agent will behave as if
        the action space is directly encoded the way it wants.

    """
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
                if isinstance(action_space_converter._template_act, self.init_action_space.template_act):
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
            self.action_space.init_converter(**kwargs_converter)

    def convert_obs(self, observation):
        """
        This function convert the observation, that is an object of class :class:`grid2op.Observation.Observation`
        into a representation understandable by the Agent.

        For example, and agent could only want to look at the relative flows :attr:`grid2op.Observation.Observation.rho`
        to take his decision. This is possible by  overloading this method.

        This method can also be used to scale the observation such that each compononents has mean 0 and variance 1
        for example.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            Initial observation received by the agent in the :func:`Agent.act` method.

        Returns
        -------
        res: ``object``
            Anything that will be used by the Agent to take decisions.

        """
        return self.action_space.convert_obs(observation)

    def convert_act(self, encoded_act):
        """
        This function will convert an "ecnoded action" that be of any types, to a valid action that can be ingested
        by the environment.

        Parameters
        ----------
        encoded_act: ``object``
            Anything that represents an action.

        Returns
        -------
        act: :grid2op.Action.Action`
            A valid actions, represented as a class, that corresponds to the encoded action given as input.

        """
        return self.action_space.convert_act(encoded_act)

    def act(self, observation, reward, done=False):
        """
        Standard method of an :class:`Agent`. There is no need to overload this function.

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
        transformed_observation = self.convert_obs(observation)
        encoded_act = self.my_act(transformed_observation, reward, done)
        return self.convert_act(encoded_act)

    @abstractmethod
    def my_act(self, transformed_observation, reward, done=False):
        """
        This method should be overide if this class is used. It is an "abstract" method.

        If someone wants to make a agent that handles different kinds of actions an observation.

        Parameters
        ----------
        transformed_observation: ``object``
            Anything that will be used to create an action. This is the results to the call of
            :func:`AgentWithConverter.convert_obs`. This is likely a numpy array.

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: ``object``
            A representation of an action in any possible format. This action will then be ingested and formatted into
            a valid action with the :func:`AgentWithConverter.convert_act` method.

        """
        transformed_action = None
        return transformed_action


class RandomAgent(AgentWithConverter):
    """
    This agent acts randomnly on the powergrid. It uses the :class:`grid2op.Converters.IdToAct` to compute all the
    possible actions available for the environment. And then chooses a random one among all these.
    """
    def __init__(self, action_space, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)

    def my_act(self, transformed_observation, reward, done=False):
        return self.action_space.sample()


class MLAgent(AgentWithConverter):
    """
    This agent allows to handle only vectors. The "my_act" function will return "do nothing" action (so it needs
    to be override)

    In this class, the "my_act" is expected to return a vector that can be directly converted into a valid action.
    """
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

