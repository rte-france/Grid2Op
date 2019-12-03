from abc import ABC, abstractmethod
import numpy as np

try:
    from .Exceptions import Grid2OpException
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException

import pdb


class LegalAction(ABC):
    """
    This class is a base class that determines whether or not an action is legal in certain environment.
    See the definition of :func:`LegalAction.__call__` for more information.

    Basically, this is an empty class with an overload of the __call__ operator that should return ``True`` or ``False``
    depending on the legality of the action.

    In :class:`grid2op.Environment`, only action of the users are checked for legality.

    """
    @abstractmethod
    def __call__(self, action, env):
        """
        As opposed to "ambiguous action", "illegal action" are not illegal per se.
        They are legal or not on a certain environment. For example, disconnecting
        a powerline that has been cut off for _maintenance is illegal. Saying to action to both disconnect a
        powerline and assign it to bus 2 on it's origin end is ambiguous, and not tolerated in Grid2Op.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action of which the legality is tested.
        env: :class:`grid2op.Environment.Environment`
            The environment on which the action is performed.

        Returns
        -------
        res: ``bool``
            Whether the action is legal or not

        """
        pass


class AllwaysLegal(LegalAction):
    """
    This subclass doesn't implement any rules regarding the legality of the actions. All actions are legal.

    """
    def __call__(self, action, env):
     """
     All actions being legal, this returns always true.
     See :func:`LegalAction.__call__` for a definition of the parameters of this function.

     """
     return True


class LookParam(LegalAction):
    """
    This subclass only check that the number of powerlines reconnected / disconnected by the agent.

    This class doesn't require any environment information. The "env" argument is only used to look for the
    game rules implemented in :class:`grid2op.Parameters`.

    See :func:`LegalAction.__call__` for a definition of the parameters of this function.

    """
    def __call__(self, action, env):
        """
        See :func:`LegalAction.__call__` for a definition of the parameters of this function.
        """
        aff_lines, aff_subs = action.get_topological_impact()
        if np.sum(aff_lines) > env.parameters.MAX_LINE_STATUS_CHANGED:
            return False
        if np.sum(aff_subs) > env.parameters.MAX_SUB_CHANGED:
            return False
        return True


class PreventReconection(LegalAction):
    """
    A subclass is used to check that an action will not attempt to reconnect a powerlines disconnected because of
    an overflow, or to check that 2 actions acting on the same powerline are distant from the right number of timesteps
    (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF`) or if two topological modification
    of the same substation are too close in time
    (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_TOPOLOGY_REMODIF`)

    """
    def __call__(self, action, env):
        """
        This function check only that the action doesn't attempt to reconnect  a powerline that has been disconnected
        due to an overflow.

        See :func:`LegalAction.__call__` for a definition of the parameters of this function.

        """
        aff_lines, aff_subs = action.get_topological_impact()
        if np.any(env.time_remaining_before_reconnection[aff_lines] > 0):
            # i tried to act on a powerline removed because an overflow
            return False

        if np.any(env.times_before_line_status_actionable[aff_lines] > 0):
            # i tried to act on a powerline too shortly after a previous action
            return False

        if np.any(env.times_before_topology_actionable[aff_subs] > 0):
            # I tried to act on a topology too shortly after a previous action
            return False

        return True


class DefaultRules(LookParam, PreventReconection):
    """
    This subclass combine both :class:`LookParam` and :class:`PreventReconection`.
    An action is declared legal if and only if:

      - It doesn't diconnect / reconnect more power lines than  what stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to act on more substations that what is stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to reconnect a powerline out of service.

    """
    def __call__(self, action, env):
        """
        See :func:`LegalAction.__call__` for a definition of the _parameters of this function.
        """
        if not LookParam.__call__(self, action, env):
            return False
        return PreventReconection.__call__(self, action, env)


class GameRules(object):
    """
    Class that defin the rules of the game.

    """
    def __init__(self, legalActClass=AllwaysLegal):
        """

        Parameters
        ----------
        legalActClass: ``type``
            The class that will be used to tell if the actions are legal or not. The class must be given, and not
            an object of this class. It should derived from :class:`LegalAction`.
        """
        if not isinstance(legalActClass, type):
            raise Grid2OpException("Parameter \"legalActClass\" used to build the GameRules should be a type (a class) and not an object (an instance of a class). It is currently \"{}\"".format(type(rewardClass)))

        if not issubclass(legalActClass, LegalAction):
            raise Grid2OpException("Gamerules: legalActClass should be initialize with a class deriving from LegalAction and not {}".format(type(legalActClass)))
        self.legal_action = legalActClass()

    def __call__(self, action, env):
        """
        Says if an action is legal or not.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action that need to be tested

        env: :class:`grid2op.Environment.Environment`
            The current used environment.

        Returns
        -------
        res: ``bool``
            Assess if the given action is legal or not. ``True``: the action is legal, ``False`` otherwise

        """
        return self.legal_action(action, env)