__all__ = [
    "BaseAction",
    "ActionSpace",
    "PowerLineSet",
    "SerializableActionSpace",
    "TopoAndRedispAction",
    "TopologyAction",
    "VoltageOnlyAction",
    'DontAct',
    "HelperAction",
    "CompleteAction",
    "Action"
]

from grid2op.Action.BaseAction import BaseAction
from grid2op.Action.ActionSpace import ActionSpace
from grid2op.Action.PowerLineSet import PowerLineSet
from grid2op.Action.SerializableActionSpace import SerializableActionSpace
from grid2op.Action.TopoAndRedispAction import TopoAndRedispAction
from grid2op.Action.TopologyAction import TopologyAction
from grid2op.Action.VoltageOnlyAction import VoltageOnlyAction
from grid2op.Action.DontAct import DontAct
import warnings


# CompleteAction to be symetrical to CompleteObservation
CompleteAction = BaseAction


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
