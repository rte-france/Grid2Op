# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Dict, Literal, Any, Union, List
import numpy as np

#: type hints corresponding to the "info" part of the env.step return value
STEP_INFO_TYPING = Dict[Literal["disc_lines",
                                "is_illegal",
                                "is_ambiguous",
                                "is_dispatching_illegal",
                                "is_illegal_reco",
                                "reason_alarm_illegal",
                                "reason_alert_illegal",
                                "opponent_attack_line",
                                "opponent_attack_sub",
                                "exception",
                                "detailed_infos_for_cascading_failures",
                                "rewards",
                                "time_series_id"],
                        Any]

#: Dict representing an action
DICT_ACT_TYPING = Dict[Literal["set_line_status",
                               "change_line_status",
                               "set_bus", 
                               "change_bus",
                               "redispatch",
                               "set_storage",
                               "curtail",
                               "raise_alarm",
                               "raise_alert",
                               "injection",
                               "hazards",
                               "maintenance",
                               "shunt"],
                       Any]
# TODO improve that (especially the Any part)

#: type hints for the "options" flag of reset function
RESET_OPTIONS_TYPING = Union[Dict[Literal["time serie id"], int],
                             Dict[Literal["init state"], DICT_ACT_TYPING],
                             Dict[Literal["init ts"], int],
                             Dict[Literal["max step"], int],
                             None]

#: type hints for a "GridObject" when converted to a dictionary
CLS_AS_DICT_TYPING = Dict[str,
                          Union[int,  # eg n_sub, or n_line
                                str,  # eg name_shunt, name_load
                                np.ndarray,  # eg load_to_subid, gen_pos_topo_vect
                                List[Union[int, str, float, bool]]]
                          ]

#: n_busbar_per_sub
N_BUSBAR_PER_SUB_TYPING = Union[int,           # one for all substation
                                List[int],     # give info for all substations
                                Dict[str, int] # give information for some substation
                                ]
