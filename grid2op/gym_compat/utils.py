# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

_MIN_GYM_VERSION = "0.17.2"

ALL_ATTR = ("set_line_status",
            "change_line_status",
            "set_bus",
            "change_bus",
            "redispatch",
            "set_storage",
            "curtail",
            "raise_alarm")

ATTR_DISCRETE = ("set_line_status",
                 "change_line_status",
                 "set_bus",
                 "change_bus",
                 "sub_set_bus",
                 "sub_change_bus",
                 "one_sub_set",
                 "one_sub_change",
                 "raise_alarm")


def check_gym_version():
    import gym
    if gym.__version__ < _MIN_GYM_VERSION:
        raise RuntimeError(f"Grid2op does not work with gym < {_MIN_GYM_VERSION} and you have gym with "
                           f"version {gym.__version__} installed.")


def _compute_extra_power_for_losses(gridobj):
    """
    to handle the "because of the power losses gen_pmin and gen_pmax can be slightly altered"
    """
    import numpy as np
    return 0.01 * np.sum(np.abs(gridobj.gen_pmax))