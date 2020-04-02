from grid2op.Exceptions.Grid2OpException import Grid2OpException


# BaseObservation
# Functionality not implemented by the observation
class NoForecastAvailable(Grid2OpException):
    """
    This exception is mainly raised by the :class:`grid2op.BaseObservation`. It specifies the :class:`grid2op.Agent.Agent`
    that the :class:`grid2op.ChronicsHandler.GridValue` doesn't produce any forecasts.

    In that case it is not possible to use the :func:`grid2op.BaseObservation.BaseObservation.forecasts` method.
    """
    pass
