# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.grid2OpException import Grid2OpException


class BaseObservationError(Grid2OpException):
    """
    Generic type of exceptions raised by the observation
    """

    pass


# BaseObservation
# Functionality not implemented by the observation
class NoForecastAvailable(Grid2OpException):
    """
    This exception is mainly raised by the :class:`grid2op.Observation.BaseObservation`. It specifies the
    :class:`grid2op.Agent.BaseAgent`
    that the :class:`grid2op.Chronics.GridValue` doesn't produce any forecasts.

    In that case it is not possible to use the :func:`grid2op.Observation.BaseObservation.forecasts` method.
    """

    pass

class SimulateError(Grid2OpException):
    """Generic error concerning the `obs.simulate(...)` function"""
    pass

<<<<<<< HEAD
class SimulateUsedTooMuch(SimulateError):
    """More precise error: you called `obs.simulate(...)` too much, raising an error"""
=======
class SimulateError(BaseObservationError):
    """
    This is the generic exception related to :func:`grid2op.Observation.BaseObservation.simulate` function
    """

    pass


class SimulateUsedTooMuch(SimulateError):
>>>>>>> bd_dev
    pass


class SimulateUsedTooMuchThisStep(SimulateUsedTooMuch):
    """
    This exception is raised by the :class:`grid2op.Observation.BaseObservation` when using "obs.simulate(...)".

    It is raised when the total number of calls to `obs.simulate(...)` exceeds the maximum number of allowed
    calls to it, for a given step.

    You can do more "obs.simulate(...)" at the next observation (after calling "env.step(...)").
    """

    pass


class SimulateUsedTooMuchThisEpisode(SimulateUsedTooMuch):
    """
    This exception is raised by the :class:`grid2op.Observation.BaseObservation` when using "obs.simulate(...)".

    It is raised when the total number of calls to `obs.simulate(...)` exceeds the maximum number of allowed
    calls to it for this episode.

    The only way to use "obs.simulate(...)" again is to call "env.reset(...)"
    """

    pass
