__all__ = [
    # Internals
    "BaseAction",
    "ActionSpace",
    "SerializableActionSpace",
    "PlayableAction",
    "VoltageOnlyAction",
    "CompleteAction",
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

import warnings

# CompleteAction to be symetrical to CompleteObservation
CompleteAction = BaseAction


class TopoAndRedispAction(TopologyAndDispatchAction):
    def __init__(self, gridobj):
        super().__init__(self, gridobj)
        warnings.warn("TopoAndRedispAction has been renamed to TopologyAndDispatchAction"
                      " -- The old name will be removed in future versions",
                      category=PendingDeprecationWarning)
    

class HelperAction(ActionSpace):
    def __init__(self, *args, **kwargs):
        ActionSpace.__init__(self, *args, **kwargs)
        warnings.warn("HelperAction class has been renamed \"ActionSpace\" to be better integrated with "
                      "openai gym framework. The old name will be removed in future "
                      "versions.",
                      category=PendingDeprecationWarning)


class Action(BaseAction):
    def __init__(self, *args, **kwargs):
        BaseAction.__init__(self, *args, **kwargs)
        warnings.warn("Action class has been renamed \"BaseAction\" if it was the Base class of each Action class,"
                      "or \"CompleteAction\" for the action that gives the possibility to do every grid "
                      "manipulation in grid2op. This class Action will be removed in future versions.",
                      category=PendingDeprecationWarning)
