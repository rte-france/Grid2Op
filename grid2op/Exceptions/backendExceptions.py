# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.grid2OpException import Grid2OpException


# Backend
class BackendError(Grid2OpException):
    """
    Base class of all error regarding the Backend that might be badly configured.
    """

    pass


class DivergingPowerflow(BackendError):
    """
    This exception indicate that the :class:`grid2op.Backend.Backend` is not able to find a valid solution to the
     physical _grid it represents.

    This divergence can be due to:

      - the system is not feasible: there is no solution to Kirchhoff's law given the state
      - the powergrid is not connected and some area of the grid do not have slack buses
      - there is a "voltage collapse" : the voltages are ill conditioned making the _grid un realistic.
      - the method to solve the powerflow fails to find a valid solution. In this case, adopting a different
        :class:`grid2op.Backend.Backend` might solve the problem.
    """
    pass


class IslandedGrid(BackendError):
    """Specific error when then backend "fails" because of an islanded grid"""
    pass


class IsolatedElement(IslandedGrid):
    """Specific error that should be raised when a element is alone on a bus (islanded grid when only one element is islanded)
    """
    pass


class DisconnectedLoad(BackendError):
    """Specific error raised by the backend when a load is disconnected"""
    pass


class DisconnectedGenerator(BackendError):
    """Specific error raised by the backend when a generator is disconnected"""
    pass


class ImpossibleTopology(BackendError):
    """Specific error raised by the backend :func:`grid2op.Backend.Backend.apply_action`
    when the player asked a topology (for example using `set_bus`) that 
    cannot be applied by the backend.
    """