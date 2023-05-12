# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np

from packaging import version
from importlib.metadata import distribution
NUMPY_VERSION = version.parse(distribution('numpy').version)
_MAX_NUMPY_VERSION_NPINT = version.parse("1.20.0") 

dt_int = (
    np.int32
)  # dtype('int64') or dtype('int32') depending on platform => i force it to int32
dt_float = (
    np.float32
)  # dtype('float64') or dtype('float32') depending on platform  => i force it to float32

dt_bool = np.bool_  # mandatory for numpy >= 1.24

int_types = (int, dt_int, np.int64)
if NUMPY_VERSION <= _MAX_NUMPY_VERSION_NPINT:
    int_types = (int, dt_int, np.int64, np.int)
