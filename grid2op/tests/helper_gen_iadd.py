#!/usr/bin/env python3

import itertools

header_content = """
import unittest
import warnings
from abc import ABC, abstractmethod
import numpy as np

import grid2op
from grid2op.Action import *
from grid2op.dtypes import dt_float
"""
print (header_content)

base_content = """
class Test_iadd_Base(ABC):

    @abstractmethod
    def _action_setup(self):
        pass

    def _skipMissingKey(self, key):
        if key not in self.action_t.authorized_keys:
            skip_msg = "Skipped: Missing authorized_key {key}"
            unittest.TestCase.skipTest(self, skip_msg)

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.action_t = cls._action_setup()
            cls.env = grid2op.make("rte_case14_realistic",
                                   test=True,
                                   action_class=cls.action_t)
    @classmethod        
    def tearDownClass(cls):
        cls.env.close()
"""
print (base_content)

actions_names = [
    "dn",
    "set_line",
    "change_line",
    "set_bus",
    "change_bus",
    "redisp"
]

actions_skip = [
    """        # No skip for do nothing""",    
    """        self._skipMissingKey("set_line_status")""",
    """        self._skipMissingKey("change_line_status")""",
    """        self._skipMissingKey("set_bus")""",
    """        self._skipMissingKey("change_bus")""",
    """        self._skipMissingKey("redispatch")"""
]

actions = [
    """self.env.action_space({})""",
    
    """self.env.action_space({
            "set_line_status": [(0, -1)]
        })""",
    
    """self.env.action_space({
            "change_line_status": [0]
        })""",
    
    """self.env.action_space({
            "set_bus": {
                "substations_id": [
                    (0, [2] + [0] * (self.env.sub_info[0] - 1))
                ]
            }
        })""",
    
    """self.env.action_space({
            "change_bus": {
                "substations_id": [
                    (0, [True] + [False] * (self.env.sub_info[0] - 1))
                ]
            }
        })""",
    
    """self.env.action_space({
            "redispatch": {
                "2": 1.42
            }
        })"""
]

actions_test = [
    """
        assert np.all({1}._set_line_status == 0)
        assert np.all({1}._switch_line_status == False)
        assert np.all({1}._set_topo_vect == 0)
        assert np.all({1}._change_bus_vect == 0)
        assert np.all({1}._redispatch == 0.0)
    """,
    
    """
        assert {1}._set_line_status[0] == -1
        assert np.all({1}._set_line_status[1:] == 0)
        assert np.all({1}._switch_line_status == False)
        assert np.all({1}._set_topo_vect == 0)
        assert np.all({1}._change_bus_vect == 0)
        assert np.all({1}._redispatch == 0.0)
    """,
    
    """
        assert np.all({1}._set_line_status == 0)
        assert {1}._switch_line_status[0] == True
        assert np.all({1}._switch_line_status[1:] == False)
        assert np.all({1}._set_topo_vect == 0)
        assert np.all({1}._change_bus_vect == 0)
        assert np.all({1}._redispatch == 0.0)
    """,
    
    """
        assert np.all({1}._set_line_status == 0)
        assert np.all({1}._switch_line_status == False)
        assert {1}._set_topo_vect[0] == 2
        assert np.all({1}._set_topo_vect[1:] == 0)
        assert np.all({1}._change_bus_vect == 0)
        assert np.all({1}._redispatch == 0.0)
    """,
    
    """
        assert np.all({1}._set_line_status == 0)
        assert np.all({1}._switch_line_status == False)
        assert np.all({1}._set_topo_vect == 0)
        assert {1}._change_bus_vect[0] == True
        assert np.all({1}._change_bus_vect[1:] == False)
        assert np.all({1}._redispatch == 0.0)
    """,
    
    """
        assert np.all({1}._set_line_status == 0)
        assert np.all({1}._switch_line_status == False)
        assert np.all({1}._set_topo_vect == 0)
        assert np.all({1}._change_bus_vect == 0)
        assert {1}._redispatch[2] == dt_float(1.42)
        assert np.all({1}._redispatch[:2] == 0.0)
        assert np.all({1}._redispatch[3:] == 0.0)
    """
]

fn_name = "    def test_{0}_iadd_{1}(self):"

a_create = """
        # Create action me [{0}]
        act_me = {1}
"""
a_test = "        # Test action me [{0}]"

b_create = """        # Create action oth [{0}]
        act_oth = {1}
"""
b_test = "        # Test action oth [{0}]"

iadd_content = """        # Iadd actions
        act_me += act_oth
"""
test_content = """        # Test combination:
        assert False, "TODO {} += {} test dumdumb"
"""

for c in itertools.product(range(len(actions)), repeat=2):
    a_idx = c[0]
    b_idx = c[1]
    a_skip = actions_skip[a_idx]
    b_skip = actions_skip[b_idx]
    a_name = actions_names[a_idx]
    b_name = actions_names[b_idx]
    a_act = actions[a_idx]
    b_act = actions[b_idx]
    a_t = actions_test[a_idx]
    b_t = actions_test[b_idx]

    print(fn_name.format(a_name, b_name))
    if len(a_skip) > 0:
        print(a_skip)
    if len(b_skip) > 0:
        print(b_skip)
    print(a_create.format(a_name, a_act))
    print((a_test + a_t).format(a_name, "act_me"))
    print(b_create.format(b_name, b_act))
    print((b_test + b_t).format(b_name, "act_oth"))
    print(iadd_content)
    print(test_content.format(a_name, b_name))


classes_names = [
    "CompleteAction",
    "DispatchAction",
    "DontAct",
    "PlayableAction",
    "PowerlineChangeAction",
    "PowerlineChangeAndDispatchAction",
    "PowerlineSetAction",
    "PowerlineSetAndDispatchAction",
    "TopologyAction",
    "TopologyAndDispatchAction",
    "TopologyChangeAction",
    "TopologyChangeAndDispatchAction",
    "TopologySetAction",
    "TopologySetAndDispatchAction"
]

class_suite = """
class Test_iadd_{0}(Test_iadd_Base, unittest.TestCase):
    \"""
    Action iadd test suite for subclass: {0}
    \"""

    @classmethod
    def _action_setup(self):
        return {0}
"""

for c_name in classes_names:
    print (class_suite.format(c_name))


main_content = """
if __name__ == "__main__":
    unittest.main()"""

print (main_content)
