from grid2op.Exceptions.Grid2OpException import Grid2OpException


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
