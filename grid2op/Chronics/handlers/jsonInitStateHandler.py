# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Union
import json
import os

import grid2op
from grid2op.Exceptions import Grid2OpException
from grid2op.Chronics.handlers.baseHandler import BaseHandler


class JSONInitStateHandler(BaseHandler):
    """Base class to initialize the grid state using a method in the time series.
    
    .. versionadded:: 1.10.2
    
    This class will look for a file named "init_state.json" (located at `self.path`) which should be a valid
    json file (you can load it with the `json` module) representing a valid
    action on the grid.
    
    This action should preferably be using only the `set` (*eg* `set_bus` and `set_status`)
    keyword arguments and set only the topology of the grid (and not the injection or
    the redispatching.)
    
    If no "init_state.json" file is found, then nothing is done.
    
    """
    
    def check_validity(self, backend) -> None:
        """This type of handler is always valid."""
        pass
    
    def done(self) -> bool:
        return False
    
    def get_init_dict_action(self) -> Union[dict, None]:
        maybe_path = os.path.join(self.path, "init_state.json")
        if not os.path.exists(maybe_path):
            return None
        try:
            with open(maybe_path, "r", encoding="utf-8") as f:
                maybe_act_dict = json.load(f)
        except Exception as exc_:
            raise Grid2OpException(f"Invalid action provided to initialize the powergrid (not readable by json)."
                                   f"Check file located at {maybe_path}") from exc_
        return maybe_act_dict
