# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Any, Optional, Dict, Tuple
from getting_started.grid2op.Observation.baseObservation import BaseObservation
from getting_started.grid2op.Observation.observationSpace import ObservationSpace
from grid2op.Action.ActionSpace import ActionSpace

from grid2op.Action.BaseAction import BaseAction

AgentID = str

LocalObservationSpace = ObservationSpace
LocalActionSpace = ActionSpace

LocalAction = BaseAction
LocalObservation = BaseObservation

ActionProfile = Dict[AgentID, BaseAction]
MADict = Dict[AgentID, Any]  # TODO slit it in more things to be more precise
# this is close to useless this way
