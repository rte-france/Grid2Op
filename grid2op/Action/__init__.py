__all__ = [
    "Action",
    "ActionSpace",
    "PowerLineSet",
    "SerializableActionSpace",
    "TopoAndRedispAction",
    "TopologyAction",
    "VoltageOnlyAction"
]

from grid2op.Action.Action import Action
from grid2op.Action.ActionSpace import ActionSpace
from grid2op.Action.PowerLineSet import PowerLineSet
from grid2op.Action.SerializableActionSpace import SerializableActionSpace
from grid2op.Action.TopoAndRedispAction import TopoAndRedispAction
from grid2op.Action.TopologyAction import TopologyAction
from grid2op.Action.VoltageOnlyAction import VoltageOnlyAction
import warnings


class HelperAction(ActionSpace):
    def __init__(self, *args, **kwargs):
        ActionSpace.__init__(*args, **kwargs)
        warnings.warn("HelperAction class has been renamed \"ActionSpace\" to be better integrated with "
                      "openai gym framework. The old name will be removed in future "
                      "versions.",
                      category=PendingDeprecationWarning)
