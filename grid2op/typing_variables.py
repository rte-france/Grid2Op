# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Dict, Literal, Any, Union

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

#: type hints for the "options" flag of reset function
RESET_OPTIONS_TYPING = Union[Dict[Union[Literal["time serie id", "init state"], str], Union[int, str]], None]
