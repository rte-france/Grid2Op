__all__ = [
    # Internals
    "BaseAction",
    "ActionSpace",
    "SerializableActionSpace",
    "PlayableAction",
    "VoltageOnlyAction",
    # Usable
    "DontAct",
    "PowerlineSetAction",
    "PowerlineChangeAction",
    "TopologyAction",
    "TopologyAndDispatchAction",
    "TopologySetAction",
    "TopologySetAndDispatchAction",
    "TopologyChangeAction",
    "TopologyChangeAndDispatchAction",
    "DispatchAction",
    # Backwards compat
    "TopoAndRedispAction",
    "HelperAction",
    "Action"
]

# Internals
from grid2op.Action.BaseAction import BaseAction
from grid2op.Action.ActionSpace import ActionSpace
from grid2op.Action.SerializableActionSpace import SerializableActionSpace
from grid2op.Action.PlayableAction import PlayableAction
from grid2op.Action.VoltageOnlyAction import VoltageOnlyAction

from grid2op.Action.DontAct import DontAct
from grid2op.Action.PowerlineSetAction import PowerlineSetAction
from grid2op.Action.PowerlineChangeAction import PowerlineChangeAction
from grid2op.Action.TopologyAction import TopologyAction
from grid2op.Action.TopologyAndDispatchAction import TopologyAndDispatchAction
from grid2op.Action.TopologySetAction import TopologySetAction
from grid2op.Action.TopologySetAndDispatchAction import TopologySetAndDispatchAction
from grid2op.Action.TopologyChangeAction import TopologyChangeAction
from grid2op.Action.TopologyChangeAndDispatchAction import TopologyChangeAndDispatchAction
from grid2op.Action.DispatchAction import DispatchAction

import warnings


class TopoAndRedispAction(TopologyAndDispatchAction):
    """
    .. deprecated:: 0.7.0
        Use :class:`TopologyAndDispatchAction` instead.

    This class has been renamed :class:`TopologyAndDispatchAction` for better consistency with others.
    """
    def __init__(self, gridobj):
        super().__init__(self, gridobj)
        warnings.warn("TopoAndRedispAction has been renamed to TopologyAndDispatchAction"
                      " -- The old name will be removed in future versions",
                      category=PendingDeprecationWarning)
    

class HelperAction(ActionSpace):
    """
    .. deprecated:: 0.7.0
        Use :class:`ActionSpace` instead.

    This class has been renamed :class:`ActionSpace` for better consistency with others.
    """
    def __init__(self, *args, **kwargs):
        ActionSpace.__init__(self, *args, **kwargs)
        warnings.warn("HelperAction class has been renamed \"ActionSpace\" to be better integrated with "
                      "openai gym framework. The old name will be removed in future "
                      "versions.",
                      category=PendingDeprecationWarning)


class Action(BaseAction):
    """
    .. deprecated:: 0.7.0
        Use :class:`BaseAction` if "Action" was used to denote the base type of which all actions should inherit,
        or use :class:`CompleteAction` to give the possibility at your agent to act on everything.

    For better consistency and clarity, this class has been split in two and renamed.

    The base class of each "action" has been nmaed :class:`BaseAction`.

    For allowing participants to take any possible actions, please use the :class:`TopologyAndDispatchAction`.
    """
    def __init__(self, *args, **kwargs):
        BaseAction.__init__(self, *args, **kwargs)
        warnings.warn("Action class has been renamed \"BaseAction\" if it was the Base class of each Action class,"
                      "or \"CompleteAction\" for the action that gives the possibility to do every grid "
                      "manipulation in grid2op. This class Action will be removed in future versions.",
                      category=PendingDeprecationWarning)
