from grid2op.Exceptions.Grid2OpException import Grid2OpException


# Exception Runner is used twice, not possible on windows / macos due to the way multiprocessing works
class AgentError(Grid2OpException):
    """
    This exception indicate that there is an error in the creation of an agent
    """
    pass
