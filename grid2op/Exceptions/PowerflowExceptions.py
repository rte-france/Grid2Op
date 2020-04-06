# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

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
