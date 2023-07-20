# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from grid2op.Action.baseAction import BaseAction


class CompleteAction(BaseAction):
    """
    Class representing the possibility to play everything.

    It cannot (and should not) be used by an Agent. Indeed, Agent actions are limited to :class:`PlayableAction`. This
    class is used by the chronics, the environment the opponent or the voltage controler for example.
    """

    def __init__(self):
        BaseAction.__init__(self)
