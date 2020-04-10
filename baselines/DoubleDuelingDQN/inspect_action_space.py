#!/usr/bin/env python3

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import argparse
import json
import tensorflow as tf
import numpy as np

from grid2op.MakeEnv import make2
from grid2op.Action import *

from DoubleDuelingDQNAgent import DoubleDuelingDQNAgent

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)
def cli():
    parser = argparse.ArgumentParser(description="Action space inspector")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    return parser.parse_args()

def prune_impact_bool(impact_section, bool_key):
    if impact_section is None:
        return None
    if bool_key not in impact_section:
        return None
    if impact_section[bool_key] == False:
        return None

    return impact_section

def prune_impact_count(impact_section, count_key):
    if impact_section is None:
        return None
    if count_key not in impact_section:
        return None
    if impact_section[count_key] == 0:
        return None
    
    impact_section.pop(count_key)
    return impact_section

def prune_impact_array(impact_section, array_key):
    if impact_section is None:
        return None
    if array_key not in impact_section:
        return None
    if len(impact_section[array_key]) == 0:
        return None

    return impact_section[array_key]

def print_actions(agent):
    actions_dict = {}
    for i in range(agent.action_size):
        index_str = str(i)
        action = agent.convert_act(i)
        impact = action.impact_on_objects()
        pruned_impact = {
            "injection": prune_impact_bool(impact["injection"], "changed"),
            "line_reconnect": prune_impact_count(impact["force_line"]["reconnections"], "count"),
            "line_disconnect": prune_impact_count(impact["force_line"]["disconnections"], "count"),
            "connect_bus": prune_impact_array(impact["topology"], "assigned_bus"),
            "disconnect_bus": prune_impact_array(impact["topology"], "disconnect_bus"),
            "switch_line": prune_impact_array(impact["switch_line"], "powerlines"),
            "switch_bus": prune_impact_array(impact["topology"], "bus_switch"),
            "redispatch": prune_impact_array(impact["redispatch"], "generators")
        }
        compact_impact = {k: v for k, v in pruned_impact.items() if v is not None}
        actions_dict[index_str] = compact_impact

    actions_json = json.dumps(actions_dict,
                              indent=2,
                              cls=NpEncoder)
    print (actions_json)
        

if __name__ == "__main__":
    args = cli()
    env = make2(args.path_data, action_class=PowerlineChangeAndDispatchAction)
    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = DoubleDuelingDQNAgent(env, env.action_space,
                                  is_training=False)
    print_actions(agent)
