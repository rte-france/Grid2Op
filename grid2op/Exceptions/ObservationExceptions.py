# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.Grid2OpException import Grid2OpException


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
