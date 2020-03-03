"""
In this module are defined all the exceptions that are used in the Grid2Op package.

They all inherit from :class:`Grid2OpException`, which is nothing more than a :class:`RuntimeError` with
customs :func:`Grid2OpException.__repr__` and :func:`Grid2OpException.__str__` definition to allow for easier logging.

"""
import inspect


class Grid2OpException(RuntimeError):
    """
    Base Exception from which all Grid2Op raise exception derived.
    """
    def vect_hierarchy_cleaned(self):
        hierarchy = inspect.getmro(self.__class__)
        names_hierarchy = [el.__name__ for el in hierarchy]
        names_hierarchy = names_hierarchy[::-1]
        # i = names_hierarchy.index("RuntimeError")
        i = names_hierarchy.index("Grid2OpException")
        names_hierarchy = names_hierarchy[i:]
        res = " ".join(names_hierarchy) + " "
        return res

    def __repr__(self):
        res = self.vect_hierarchy_cleaned()
        res += RuntimeError.__repr__(self)
        return res

    def __str__(self):
        res = self.vect_hierarchy_cleaned()
        res += "\"{}\"".format(RuntimeError.__str__(self))
        return res


# Unknown environment at creating
class UnknownEnv(Grid2OpException):
    """
    This exception indicate that a bad argument has been sent to the :func:`grid2op.make` function.

    It does not recognize the name of the :class:`grid2op.Environment.Environment`.
    """
    pass


# Exception bad environment configured
class EnvError(Grid2OpException):
    """
    This exception indicate that the :class:`grid2op.Environment.Environment` is poorly configured.

    It is for example thrown when assessing if a backend is properly set up with
    :func:`grid2op.Backend.Backend.assert_grid_correct`
    """
    pass


class IncorrectNumberOfLoads(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the total number of
    loads of the powergrid.
    """
    pass


class IncorrectNumberOfGenerators(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the total number of
    generators of the powergrid.
    """
    pass


class IncorrectNumberOfLines(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the total number of
     powerlines of the powergrid.
    """
    pass


class IncorrectNumberOfSubstation(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the total
    number of substation of the powergrid.
    """
    pass


class IncorrectNumberOfElements(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the total number
    of elements of the powergrid.
    """
    pass

class IncorrectPositionOfLoads(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the number of
    loads at a substation.
    """
    pass


class IncorrectPositionOfGenerators(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the number of
    generators at a substation.
    """
    pass
class IncorrectPositionOfLines(EnvError):
    """
    This is a more precise exception than :class:`EnvError` indicating that there is a mismatch in the number of
    power lines at a substation.
    """
    pass


# exception bad actions
class IllegalAction(Grid2OpException):
    """
    This exception indicate that the :class:`grid2op.Action` is illegal.

    It is for example thrown when an :class:`grid2op.Agent` tries to perform an action against the rule.
    This is handled in :func:`grid2op.Environment.Environment.step`

    An :class:`grid2op.Action` is said to be **illegal** depending on some rules implemented in
    :func:`grid2op.Action.HelperAction.is_legal` method.
    An action can be legal in some context, but illegal in others.

    """
    pass


class OnProduction(IllegalAction):
    """
    This is a more precise exception than :class:`IllegalAction` indicating that the action is illegal due to
    setting wrong values to generators.
    """
    pass


class VSetpointModified(OnProduction):
    """
    This is a more precise exception than :class:`OnProduction` indicating that the action is illegal because the
     setpoint voltage magnitude of a production has been changed.
    """
    pass


class ActiveSetPointAbovePmax(OnProduction):
    """
    This is a more precise exception than :class:`OnProduction` indicating that the action is illegal because the
    setpoint active power of a production is set to be higher than Pmax.
    """
    pass


class ActiveSetPointBelowPmin(OnProduction):
    """
    This is a more precise exception than :class:`OnProduction` indicating that the action is illegal because the
    setpoint active power of a production is set to be lower than Pmin.
    """
    pass


class OnLoad(IllegalAction):
    """
    This is a more precise exception than :class:`IllegalAction` indicating that the action is illegal due to
    setting wrong values to loads.
    """
    pass


class OnLines(IllegalAction):
    """
    This is a more precise exception than :class:`IllegalAction` indicating that the action is illegal due to setting
     wrong values to lines (reconnection impossible, disconnection impossible etc).
    """
    pass


class InvalidReconnection(OnLines):
    """
    This is a more precise exception than :class:`OnLines` indicating that the :class:`grid2op.Agent` tried to
    reconnect a powerline illegally.
    """
    pass


# ambiguous action
class AmbiguousAction(Grid2OpException):
    """
    This exception indicate that the :class:`grid2op.Action` is ambiguous. It could be understood differently according
    to the backend used.

    Such a kind of action are forbidden in this package. These kind of exception are mainly thrown by the
    :class:`grid2op.Action.Action` in
    the :func:`grid2op.Action.update` and :func:`grid2op.Action.__call__` methods.

    As opposed to a :class:`IllegalAction` an :class:`AmbiguousAction` is forbidden for all the backend,
    in all the scenarios.

    It doesn't depend on the implemented rules.
    """
    pass


class InvalidLineStatus(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action` is
    ambiguous due to powerlines manipulation.
    """
    pass


class UnrecognizedAction(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action` is
    ambiguous due to the bad formatting of the action.
    """
    pass


class InvalidNumberOfLoads(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action`
    is ambiguous because an incorrect number of loads tries to be modified.
    """
    pass


class InvalidNumberOfGenerators(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action`
    is ambiguous because an incorrect number of generator tries to be modified.
    """
    pass


class InvalidNumberOfLines(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action`
    is ambiguous because an incorrect number of lines tries to be modified.
    """
    pass


class InvalidNumberOfObjectEnds(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action`
    is ambiguous because an incorrect number of object at a substation try to be modified.
    """
    pass


class InvalidBusStatus(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action`
    try to both "set" and "switch" some bus to which an object is connected.
    """
    pass


class InvalidRedispatching(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the :class:`grid2op.Action.Action`
    try to apply an invalid redispatching strategy.
    """
    pass


class GeneratorTurnedOnTooSoon(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that a generator has been turned on
    before gen_min_up_time time steps.
    """
    pass


class GeneratorTurnedOffTooSoon(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that a generator has been turned off
    before gen_min_down_time time steps.
    """
    pass


class NotEnoughGenerators(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that there is not enough turned off
    generators to meet the demand.
    """
    pass

# powerflow exception
class DivergingPowerFlow(Grid2OpException):
    """
    This exception indicate that the :class:`grid2op.Backend.Backend` is not able to find a valid solution to the
     physical _grid it represents.

    This divergence can be due to:

      - the system is not feasible: there is no solution to Kirchhoff's law given the state
      - the powergrid is not connex
      - there is a "voltage collapse" : the voltages are ill conditioned making the _grid un realistic.
      - the method to solve the powerflow fails to find a valid solution. In this case, adopting a different
        :class:`grid2op.Backend.Backend` might solve the problem.
    """
    pass


# Observation
# Functionality not implemented by the observation
class NoForecastAvailable(Grid2OpException):
    """
    This exception is mainly raised by the :class:`grid2op.Observation`. It specifies the :class:`grid2op.Agent.Agent`
    that the :class:`grid2op.ChronicsHandler.GridValue` doesn't produce any forecasts.

    In that case it is not possible to use the :func:`grid2op.Observation.Observation.forecasts` method.
    """
    pass

# Chronics
class ChronicsError(Grid2OpException):
    """
    Base class of all error regarding the chronics and the gridValue (see :class:`grid2op.ChronicsHandler.GridValue` for
    more information)
    """
    pass

class ChronicsNotFoundError(ChronicsError):
    """
    This exception is raised where there are no chronics folder found at the indicated location.
    """
    pass

class InsufficientData(ChronicsError):
    """
    This exception is raised where there are not enough data compare to the size of the episode asked.
    """
    pass

# Backend
class BackendError(Grid2OpException):
    """
    Base class of all error regarding the Backend that might be badly configured.
    """
    pass

# attempt to use redispatching or unit commit method in an environment not set up.
class UnitCommitorRedispachingNotAvailable(Grid2OpException):
    """
    attempt to use redispatching or unit commit method in an environment not set up.
    """
    pass

# multi environment
class MultiEnvException(Grid2OpException):
    """General exception raised by :class:`grid2Op.MultiEnv.MultiEnvironment` """
    pass

# plot error
class PlotError(Grid2OpException):
    """General exception raised by any class that handles plots"""
    pass