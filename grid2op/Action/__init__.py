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
    "CompleteAction"
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
        ActionSpace.__init__(*args, **kwargs)
        warnings.warn("HelperAction class has been renamed \"ActionSpace\" to be better integrated with "
                      "openai gym framework. The old name will be removed in future "
                      "versions.",
                      category=PendingDeprecationWarning)

