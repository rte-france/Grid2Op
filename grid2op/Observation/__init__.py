__all__ = [
    "CompleteObservation",
    "_ObsEnv",
    "BaseObservation",
    "ObservationHelper",
    "Observation"
]


from grid2op.Observation.CompleteObservation import CompleteObservation
from grid2op.Observation._ObsEnv import _ObsEnv
from grid2op.Observation.BaseObservation import BaseObservation
from grid2op.Observation.ObservationSpace import ObservationSpace
import warnings


class ObservationHelper(ObservationSpace):
    def __init__(self, *args, **kwargs):
        ObservationSpace.__init__(self, *args, **kwargs)
        warnings.warn("ObservationHelper class has been renamed \"ObservationSpace\" to be better integrated with "
                      "openai gym framework. The old name will be removed in future "
                      "versions.",
                      category=PendingDeprecationWarning)


class Observation(BaseObservation):
    def __init__(self, *args, **kwargs):
        BaseObservation.__init__(self, *args, **kwargs)
        warnings.warn("Observation class has been renamed \"BaseObservation\". The Observation class will be removed"
                      "in future versions.",
                      category=PendingDeprecationWarning)