# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# do some generic tests that can be implemented directly to test if a backend implementation can work out of the box
# with grid2op.
# see an example of test_Pandapower for how to use this suit.
import unittest
import numpy as np
import warnings

import grid2op
from grid2op.Runner import Runner
from grid2op.Agent import RandomAgent, DoNothingAgent
from grid2op.Backend import PandaPowerBackend

import pdb


class TestSuitePandaPowerBackend(PandaPowerBackend):
    """Only work for the case 14 !!!"""

    def __init__(self, detailed_infos_for_cascading_failures=False,
                 can_be_copied=True,
                 **kwargs):
        PandaPowerBackend.__init__(
            self,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
            can_be_copied=can_be_copied,
            **kwargs
        )
        # just for the test
        self._nb_bus_before_for_test = 14
        self._nb_line_for_test = 15

    def apply_action(self, backendAction=None):
        # to test the get_loads_bus, get_gen_bus, etc. function
        if backendAction is None:
            return

        # TODO
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            _,
            shunts__,
        ) = backendAction()

        tmp_prod_p = self._get_vector_inj["prod_p"](self._grid)
        if np.any(prod_p.changed):
            tmp_prod_p.iloc[prod_p.changed] = prod_p.values[prod_p.changed]

        tmp_prod_v = self._get_vector_inj["prod_v"](self._grid)
        if np.any(prod_v.changed):
            tmp_prod_v.iloc[prod_v.changed] = (
                prod_v.values[prod_v.changed] / self.prod_pu_to_kv[prod_v.changed]
            )

        if self._id_bus_added is not None and prod_v.changed[self._id_bus_added]:
            # handling of the slack bus, where "2" generators are present.
            self._grid["ext_grid"]["vm_pu"] = 1.0 * tmp_prod_v[self._id_bus_added]

        tmp_load_p = self._get_vector_inj["load_p"](self._grid)
        if np.any(load_p.changed):
            tmp_load_p.iloc[load_p.changed] = load_p.values[load_p.changed]

        tmp_load_q = self._get_vector_inj["load_q"](self._grid)
        if np.any(load_q.changed):
            tmp_load_q.iloc[load_q.changed] = load_q.values[load_q.changed]

        if self.shunts_data_available:
            shunt_p, shunt_q, shunt_bus = shunts__

            if np.any(shunt_p.changed):
                self._grid.shunt["p_mw"].iloc[shunt_p.changed] = shunt_p.values[
                    shunt_p.changed
                ]
            if np.any(shunt_q.changed):
                self._grid.shunt["q_mvar"].iloc[shunt_q.changed] = shunt_q.values[
                    shunt_q.changed
                ]
            if np.any(shunt_bus.changed):
                sh_service = shunt_bus.values[shunt_bus.changed] != -1
                self._grid.shunt["in_service"].iloc[shunt_bus.changed] = sh_service
                sh_bus1 = np.arange(len(shunt_bus))[
                    shunt_bus.changed & shunt_bus.values == 1
                ]
                sh_bus2 = np.arange(len(shunt_bus))[
                    shunt_bus.changed & shunt_bus.values == 2
                ]
                if len(sh_bus1) > 0:
                    self._grid.shunt["bus"][sh_bus1] = self.shunt_to_subid[sh_bus1]
                if len(sh_bus2) > 0:
                    self._grid.shunt["bus"][sh_bus2] = (
                        self.shunt_to_subid[sh_bus2] + self._nb_bus_before_for_test
                    )

        # i made at least a real change, so i implement it in the backend
        loads_bus = backendAction.get_loads_bus()
        for load_id, new_bus in loads_bus:
            if new_bus == -1:
                self._grid.load["in_service"][load_id] = False
            else:
                self._grid.load["in_service"][load_id] = True
                self._grid.load["bus"][load_id] = (
                    self.load_to_subid[load_id]
                    + (new_bus - 1) * self._nb_bus_before_for_test
                )

        gens_bus = backendAction.get_gens_bus()
        for gen_id, new_bus in gens_bus:
            if new_bus == -1:
                self._grid.gen["in_service"][gen_id] = False
            else:
                self._grid.gen["in_service"][gen_id] = True
                self._grid.gen["bus"][gen_id] = (
                    self.gen_to_subid[gen_id]
                    + (new_bus - 1) * self._nb_bus_before_for_test
                )

                if (
                    gen_id == (self._grid.gen.shape[0] - 1)
                    and self._iref_slack is not None
                ):
                    self._grid.ext_grid["bus"].iat[0] = (
                        self.gen_to_subid[gen_id]
                        + (new_bus - 1) * self._nb_bus_before_for_test
                    )

        lines_or_bus = backendAction.get_lines_or_bus()
        for line_id, new_bus in lines_or_bus:
            if line_id < self._nb_line_for_test:
                dt = self._grid.line
                key = "from_bus"
                line_id_db = line_id
            else:
                dt = self._grid.trafo
                key = "hv_bus"
                line_id_db = line_id - self._nb_line_for_test

            if new_bus == -1:
                dt["in_service"][line_id_db] = False
            else:
                dt["in_service"][line_id_db] = True
                dt[key][line_id_db] = (
                    self.line_or_to_subid[line_id]
                    + (new_bus - 1) * self._nb_bus_before_for_test
                )

        lines_ex_bus = backendAction.get_lines_ex_bus()
        for line_id, new_bus in lines_ex_bus:
            if line_id < self._nb_line_for_test:
                dt = self._grid.line
                key = "to_bus"
                line_id_db = line_id
            else:
                dt = self._grid.trafo
                key = "lv_bus"
                line_id_db = line_id - self._nb_line_for_test

            if new_bus == -1:
                dt["in_service"][line_id_db] = False
            else:
                dt["in_service"][line_id_db] = True
                dt[key][line_id_db] = (
                    self.line_ex_to_subid[line_id]
                    + (new_bus - 1) * self._nb_bus_before_for_test
                )

        bus_is = self._grid.bus["in_service"]
        for i, (bus1_status, bus2_status) in enumerate(active_bus):
            bus_is[i] = bus1_status  # no iloc for bus, don't ask me why please :-/
            bus_is[i + self._nb_bus_before_for_test] = bus2_status


class TestXXXBus(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.envtest = grid2op.make(
                "rte_case14_realistic",
                test=True,
                backend=TestSuitePandaPowerBackend(),
                _add_to_name=type(self).__name__+"test_get_xxx_bus_test",
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.envref = grid2op.make(
                "rte_case14_realistic",
                test=True,
                _add_to_name=type(self).__name__+"test_get_xxx_bus_ref"
            )
        seed = 0
        self.nb_test = 10
        self.max_iter = 30

        self.envref.seed(seed)
        self.envtest.seed(seed)
        self.seeds = [
            i for i in range(self.nb_test)
        ]  # used for seeding environment and agent

    def tearDown(self) -> None:
        self.envref.close()
        self.envtest.close()

    def test_get_load_bus(self):
        """
        test the methods get_load_bus works
        """
        act_load = self.envref.action_space(
            {"set_bus": {"loads_id": [(0, 2)], "lines_or_id": [(2, 2)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_load)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_load)
        assert not done_ref
        assert obs_test.topo_vect[self.envtest.load_pos_topo_vect[0]] == 2
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

    def test_tricky_action(self):
        """
        do an action on the generator where the slack bus is, it "broke" the "code" i used for testing
        (TestSuitePandaPowerBackend)
        """
        act_load = self.envref.action_space(
            {"set_bus": {"generators_id": [(4, 2)], "lines_or_id": [(0, 2), (1, 1)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_load)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_load)
        assert not done_ref
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

    def test_get_gen_bus(self):
        """
        test the methods get_gen_bus works
        """
        act_gen = self.envref.action_space(
            {"set_bus": {"generators_id": [(0, 2)], "lines_or_id": [(2, 2)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_gen)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_gen)

        assert not done_ref
        assert obs_test.topo_vect[self.envtest.gen_pos_topo_vect[0]] == 2
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

    def test_get_line_or_bus(self):
        """
        test the methods get_line_or_bus works (for line and transformer)
        """
        act_gen = self.envref.action_space(
            {"set_bus": {"generators_id": [(0, 2)], "lines_or_id": [(2, 2)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_gen)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_gen)

        assert not done_ref
        assert obs_test.topo_vect[self.envtest.line_or_pos_topo_vect[2]] == 2
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

        # for trafo
        self.envref.reset()
        self.envtest.reset()
        act_gen = self.envref.action_space(
            {"set_bus": {"lines_ex_id": [(3, 2)], "lines_or_id": [(16, 2)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_gen)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_gen)

        assert not done_ref
        assert obs_test.topo_vect[self.envtest.line_or_pos_topo_vect[16]] == 2
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

    def test_get_line_ex_bus(self):
        """
        test the methods get_line_or_bus works (for line and transformer)
        """
        act_gen = self.envref.action_space(
            {"set_bus": {"lines_ex_id": [(0, 2)], "lines_or_id": [(2, 2)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_gen)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_gen)

        assert not done_ref
        assert obs_test.topo_vect[self.envtest.line_ex_pos_topo_vect[0]] == 2
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

        # for trafo
        self.envref.reset()
        self.envtest.reset()
        act_gen = self.envref.action_space(
            {"set_bus": {"lines_ex_id": [(16, 2)], "lines_or_id": [(10, 2)]}}
        )
        obs_ref, reward_ref, done_ref, info_ref = self.envref.step(act_gen)
        obs_test, reward_test, done_test, info_test = self.envtest.step(act_gen)

        assert not done_ref
        assert obs_test.topo_vect[self.envtest.line_ex_pos_topo_vect[16]] == 2
        assert obs_ref == obs_test
        assert reward_ref == reward_test
        assert done_ref == done_test

    def test_get_xxx_bus_dn(self):
        """
        test the methods get_load_bus, get_gen_bus, get_lines_or_bus and get_lines_ex_bus works with do
        nothing agent (more examples)
        """
        runner_ref = Runner(
            **self.envref.get_params_for_runner(), agentClass=DoNothingAgent
        )
        runner_test = Runner(
            **self.envtest.get_params_for_runner(), agentClass=DoNothingAgent
        )

        res_ref = runner_ref.run(
            nb_episode=self.nb_test,
            max_iter=self.max_iter,
            agent_seeds=self.seeds,
            env_seeds=self.seeds,
        )
        res_test = runner_test.run(
            nb_episode=self.nb_test,
            max_iter=self.max_iter,
            agent_seeds=self.seeds,
            env_seeds=self.seeds,
        )

        assert res_ref == res_test

    def test_get_xxx_bus_random(self):
        """
        test the methods get_load_bus, get_gen_bus, get_lines_or_bus and get_lines_ex_bus works with random
        agent (more tests)
        """
        runner_ref = Runner(
            **self.envref.get_params_for_runner(), agentClass=RandomAgent
        )
        runner_test = Runner(
            **self.envtest.get_params_for_runner(), agentClass=RandomAgent
        )

        res_ref = runner_ref.run(
            nb_episode=self.nb_test,
            max_iter=self.max_iter,
            agent_seeds=self.seeds,
            env_seeds=self.seeds,
        )
        res_test = runner_test.run(
            nb_episode=self.nb_test,
            max_iter=self.max_iter,
            agent_seeds=self.seeds,
            env_seeds=self.seeds,
        )
        assert res_ref == res_test


if __name__ == "__main__":
    unittest.main()
