from grid2op.Exceptions.Grid2OpException import Grid2OpException


# Backend
class BackendError(Grid2OpException):
    """
    Base class of all error regarding the Backend that might be badly configured.
    """
    pass
