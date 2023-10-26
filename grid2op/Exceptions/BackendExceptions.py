# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.Grid2OpException import Grid2OpException


# Backend
class BackendError(Grid2OpException):
    """
    Base class of all error regarding the Backend that might be badly configured.
    """

    pass


class DivergingPowerflow(BackendError):
    """Specific error that should be raised when the powerflow diverges
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
