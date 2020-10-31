# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


class GymObservationSpace:
    def __init__(self, env):
        raise RuntimeError("The \"GymObservationSpace\" has been moved to \"grid2op.gym\" module instead.\n"
                           "Note to update: use \"from grid2op.gym_compat import GymObservationSpace\"")


class GymActionSpace:
    def __init__(self, action_space):
        raise RuntimeError("The \"GymActionSpace\" has been moved to \"grid2op.gym\" module instead.\n"
                           "Note to update: use \"from grid2op.gym_compat import GymActionSpace\"")
