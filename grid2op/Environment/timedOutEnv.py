# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import BaseEnv

class TimedOutEnvironment(BaseEnv):
    """_summary_

    Args:
        BaseEnv (_type_): _description_
    """
    def init(self, time_out: int):
        """_summary_

        Args:
            time_out (int): _description_
        """      
        super().init()
    def step(self, action):
        for i in range(XXX):
            super().step(do_nothing)
            super().step(action)
        do_nothing_action = self.action_space()