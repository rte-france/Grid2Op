# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from packaging import version

try:
    from importlib.metadata import distribution
except ModuleNotFoundError:
    # not available in python 3.7
    from importlib_metadata import distribution

_MIN_GYM_VERSION = version.parse("0.17.2")
# this is the last gym version to use the "old" numpy prng
_MAX_GYM_VERSION_RANDINT = version.parse("0.25.99") 
# the current gym version (we should support most recent, but also 
# the very old 0.21 because it used by stable baselines3...)
GYM_VERSION = version.parse(distribution('gym').version)


ALL_ATTR = (
    "set_line_status",
    "change_line_status",
    "set_bus",
    "change_bus",
    "redispatch",
    "set_storage",
    "curtail",
    "raise_alarm",
    "raise_alert",
)

ATTR_DISCRETE = (
    "set_line_status",
    "set_line_status_simple",
    "change_line_status",
    "set_bus",
    "change_bus",
    "sub_set_bus",
    "sub_change_bus",
    "one_sub_set",
    "one_sub_change",
    "raise_alarm"
    "raise_alert"
)


def check_gym_version():

    if GYM_VERSION < _MIN_GYM_VERSION:
        import gym
        raise RuntimeError(
            f"Grid2op does not work with gym < {_MIN_GYM_VERSION} and you have gym with "
            f"version {gym.__version__} installed."
        )


def _compute_extra_power_for_losses(gridobj):
    """
    to handle the "because of the power losses gen_pmin and gen_pmax can be slightly altered"
    """
    import numpy as np

    return 0.3 * np.sum(np.abs(gridobj.gen_pmax))


def sample_seed(max_, np_random):
    """sample a seed based on gym version (np_random has not always the same behaviour)"""
    if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
        # old gym behaviour
        seed_ = np_random.randint(max_)
    else:
        # gym finally use most recent numpy random generator
        seed_ = int(np_random.integers(0, max_))
    return seed_
