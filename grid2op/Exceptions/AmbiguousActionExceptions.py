from grid2op.Exceptions.Grid2OpException import Grid2OpException

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
