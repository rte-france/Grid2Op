__all__ = [
    "CompleteObservation",
    "ObsEnv",
    "ObservationSpace",
    "BaseObservation",
    "SerializableObservationSpace",
    "ObservationHelper"
]


from grid2op.Observation.CompleteObservation import CompleteObservation
from grid2op.Observation.ObsEnv import ObsEnv
from grid2op.Observation.ObservationSpace import ObservationSpace
from grid2op.Observation.BaseObservation import BaseObservation
from grid2op.Observation.SerializableObservationSpace import SerializableObservationSpace
import warnings


class ObservationHelper(ObservationSpace):
    def __init__(self, *args, **kwargs):
        ObservationSpace.__init__(*args, **kwargs)
        warnings.warn("ObservationHelper class has been renamed \"ObservationSpace\" to be better integrated with "
                      "openai gym framework. The old name will be removed in future "
                      "versions.",
                      category=PendingDeprecationWarning)


class Observation(BaseObservation):
    def __init__(self, *args, **kwargs):
        BaseObservation.__init__(*args, **kwargs)
        warnings.warn("Observation class has been renamed \"BaseObservation\". The Observation class will be removed"
                      "in future versions.",
                      category=PendingDeprecationWarning)