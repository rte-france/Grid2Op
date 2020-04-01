from grid2op.Exceptions.Grid2OpException import Grid2OpException

# exception bad actions
class IllegalAction(Grid2OpException):
    """
    This exception indicate that the :class:`grid2op.Action` is illegal.

    It is for example thrown when an :class:`grid2op.Agent` tries to perform an action against the rule.
    This is handled in :func:`grid2op.Environment.Environment.step`

    An :class:`grid2op.Action` is said to be **illegal** depending on some rules implemented in
    :func:`grid2op.Action.ActionSpace.is_legal` method.
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

# attempt to use redispatching or unit commit method in an environment not set up.
class UnitCommitorRedispachingNotAvailable(IllegalAction):
    """
    attempt to use redispatching or unit commit method in an environment not set up.
    """
    pass
