#!/usr/bin/env python3

import unittest
import grid2op
import numpy as np
from grid2op.dtypes import dt_int

class Issue134Tester(unittest.TestCase):

    def test_issue_134(self):
        env = grid2op.make("rte_case14_realistic", test=True)
        LINE_ID = 2

        # Disconnect ex
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 0)],
                "lines_ex_id": [(LINE_ID, -1)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == False
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == -1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == -1
        
        # Reconnect ex on bus 2
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 0)],
            "lines_ex_id": [(LINE_ID, 2)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == True
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == 1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == 2
    
        # Disconnect or
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, -1)],
                "lines_ex_id": [(LINE_ID, 0)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == False
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == -1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == -1

        # Reconnect or on bus 1
        action = env.action_space({
            'set_bus': {
                "lines_or_id": [(LINE_ID, 1)],
                "lines_ex_id": [(LINE_ID, 0)],
            }
        })
        obs, reward, done, info = env.step(action)
        assert obs.line_status[LINE_ID] == True
        assert obs.topo_vect[obs.line_or_pos_topo_vect[LINE_ID]] == 1
        assert obs.topo_vect[obs.line_ex_pos_topo_vect[LINE_ID]] == 2

if __name__ == "__main__":
    unittest.main()
