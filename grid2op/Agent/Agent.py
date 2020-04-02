"""
In this RL framework, an :class:`Agent` is an entity that acts on the :class:`Environment`. Agent can alternatively
be named "bot" or "controller" in other literature.

This module presents a few possible :class:`Agent` that can serve either as baseline, or as example on how to
implement such agents.

To perform their actions, agent receive two main signals from the :class:`grid2op.Environment`:

  - the :class:`grid2op.Reward` that states how good the previous has been
  - the :class:`grid2op.BaseObservation` that is a (partial) view on the state of the Environment.

Both these signals can be use to determine what is the best action to perform on the grid. This is actually the main
objective of an :class:`Agent`, and this is done in the :func:`Agent.act` method.

"""

from abc import ABC, abstractmethod


class Agent(ABC):
    """
    This class represents the base class of an Agent. All bot / controller / agent used in the Grid2Op simulator
    should derived from this class.

    To work properly, it is advise to create Agent after the :class:`grid2op.Environment` has been created and reuse
    the :attr:`grid2op.Environment.Environment.action_space` to build the Agent.

    Attributes
    -----------
    action_space: :class:`grid2op.Action.ActionSpace`
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
