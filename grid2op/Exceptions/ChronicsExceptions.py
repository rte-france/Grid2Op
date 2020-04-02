from grid2op.Exceptions.Grid2OpException import Grid2OpException


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
