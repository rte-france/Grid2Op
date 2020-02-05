# this files present utilitary class
try:
    from .Observation import ObservationHelper as ObservationSpace
    from .Action import HelperAction as ActionSpace
except (ModuleNotFoundError, ImportError):
    from Observation import ObservationHelper as ObservationSpace
    from Action import HelperAction as ActionSpace