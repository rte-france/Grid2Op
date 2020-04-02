from abc import ABC, abstractmethod

import pdb


class BaseRules(ABC):
    """
    This class is a base class that determines whether or not an action is legal in certain environment.
    See the definition of :func:`BaseRules.__call__` for more information.

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
