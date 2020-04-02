import pdb

from grid2op.Agent.BaseAgent import BaseAgent


class DoNothingAgent(BaseAgent):
    """
    This is the most basic BaseAgent. It is purely passive, and does absolutely nothing.
    """
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)

    def act(self, observation, reward, done=False):
        """
        As better explained in the document of :func:`grid2op.BaseAction.update` or
        :func:`grid2op.BaseAction.ActionSpace.__call__`.

        The preferred way to make an object of type action is to call :func:`grid2op.BaseAction.ActionSpace.__call__` with
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
