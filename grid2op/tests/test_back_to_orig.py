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
from grid2op.Parameters import Parameters
from grid2op.Action import BaseAction
import pdb


class Test_BackToOrig(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = "educ_case14_storage"
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        param.ACTIVATE_STORAGE_LOSS = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                self.env_name, test=True, action_class=BaseAction, param=param,
                _add_to_name=type(self).__name__
            )

    def tearDown(self) -> None:
        self.env.close()

    def test_substation(self):
        obs, reward, done, info = self.env.step(
            self.env.action_space({"set_bus": {"substations_id": [(2, (1, 2, 2, 1))]}})
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {"set_bus": {"substations_id": [(5, (1, 2, 2, 1, 2, 1, 1, 2))]}}
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "substation" in res
        assert len(res["substation"]) == 2
        for act in res["substation"]:
            lines_impacted, subs_impacted = act.get_topological_impact()
            assert subs_impacted[2] ^ subs_impacted[5]  # xor
            assert np.sum(lines_impacted) == 0

        for act in res["substation"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology

    def test_line(self):
        obs = self.env.reset()
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 0
        obs, reward, done, info = self.env.step(
            self.env.action_space({"set_line_status": [(12, -1)]})
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space({"set_line_status": [(15, -1)]})
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "powerline" in res
        assert len(res["powerline"]) == 2
        for act in res["powerline"]:
            lines_impacted, subs_impacted = act.get_topological_impact()
            assert lines_impacted[12] ^ lines_impacted[15]  # xor
            assert np.sum(subs_impacted) == 0

        for act in res["powerline"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology

    def test_redisp(self):
        obs = self.env.reset()
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 0
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "redispatch": [
                        (0, self.env.gen_max_ramp_up[0]),
                        (1, -self.env.gen_max_ramp_down[1]),
                    ]
                }
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "redispatching" in res
        assert len(res["redispatching"]) == 1  # one action is enough

        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "redispatch": [
                        (0, self.env.gen_max_ramp_up[0]),
                        (1, -self.env.gen_max_ramp_down[1]),
                    ]
                }
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "redispatching" in res
        assert len(res["redispatching"]) == 2  # one action is NOT enough

        for act in res["redispatching"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            np.max(np.abs(obs.target_dispatch)) <= 1e-6
        )  # I am in the original topology
        assert len(self.env.action_space.get_back_to_ref_state(obs)) == 0

        # now try with "non integer" stuff
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "redispatch": [
                        (0, self.env.gen_max_ramp_up[0]),
                        (1, -self.env.gen_max_ramp_down[1]),
                    ]
                }
            )
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "redispatch": [
                        (0, 0.5 * self.env.gen_max_ramp_up[0]),
                        (1, -0.5 * self.env.gen_max_ramp_down[1]),
                    ]
                }
            )
        )
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "redispatching" in res
        assert len(res["redispatching"]) == 2  # one action is NOT enough
        for act in res["redispatching"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            np.max(np.abs(obs.target_dispatch)) <= 1e-6
        )  # I am in the original topology

        # try with non integer, non symmetric stuff
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "redispatch": [
                        (0, self.env.gen_max_ramp_up[0]),
                        (1, -self.env.gen_max_ramp_down[1]),
                    ]
                }
            )
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {"redispatch": [(0, 0.5 * self.env.gen_max_ramp_up[0])]}
            )
        )
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "redispatching" in res
        assert len(res["redispatching"]) == 2  # one action is NOT enough
        for act in res["redispatching"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            np.max(np.abs(obs.target_dispatch)) <= 1e-6
        )  # I am in the original topology
        assert len(self.env.action_space.get_back_to_ref_state(obs)) == 0

    def test_storage_no_loss(self):
        obs = self.env.reset()
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 0
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 1  # one action is enough (no losses)

        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 2  # one action is NOT enough

        for act in res["storage"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology

        # now try with "non integer" stuff
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, 0.5 * self.env.storage_max_p_absorb[0]),
                        (1, -0.5 * self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 2  # one action is NOT enough
        for act in res["storage"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert np.max(np.abs(obs.target_dispatch)) <= 1e-6
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology

        # try with non integer, non symmetric stuff
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {"set_storage": [(0, 0.5 * self.env.storage_max_p_absorb[0])]}
            )
        )
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 2  # one action is NOT enough
        for act in res["storage"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology

    def test_storage_with_loss(self):
        param = self.env.parameters
        param.ACTIVATE_STORAGE_LOSS = True
        self.env.change_parameters(param)
        obs = self.env.reset()
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 0

        # check i get the right power if i do nothing
        obs, reward, done, info = self.env.step(self.env.action_space())
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 1  # one action is enough to compensate the losses
        assert np.all(
            np.abs(res["storage"][0].storage_p - self.env.storage_loss) <= 1e-5
        )

        # now do some action
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 2  # one action is NOT enough (no losses)

        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 3  # two actions are NOT enough (losses)

        for act in res["storage"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        dict_ = self.env.action_space.get_back_to_ref_state(obs)
        assert len(dict_) == 1
        assert "storage" in dict_
        assert np.all(
            np.abs(dict_["storage"][0].storage_p - 3.0 * self.env.storage_loss) <= 1e-5
        )  # I am in the original topology (up to the storage losses)

        # now try with "non integer" stuff
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, 0.5 * self.env.storage_max_p_absorb[0]),
                        (1, -0.5 * self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 2  # one action is NOT enough
        for act in res["storage"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        dict_ = self.env.action_space.get_back_to_ref_state(obs)
        assert len(dict_) == 1
        assert "storage" in dict_
        assert np.all(
            np.abs(dict_["storage"][0].storage_p - 2.0 * self.env.storage_loss) <= 1e-5
        )  # I am in the original topology (up to the storage losses)

        # try with non integer, non symmetric stuff
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {
                    "set_storage": [
                        (0, self.env.storage_max_p_absorb[0]),
                        (1, -self.env.storage_max_p_prod[1]),
                    ]
                }
            )
        )
        assert not done
        obs, reward, done, info = self.env.step(
            self.env.action_space(
                {"set_storage": [(0, 0.5 * self.env.storage_max_p_absorb[0])]}
            )
        )
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "storage" in res
        assert len(res["storage"]) == 2  # one action is NOT enough
        for act in res["storage"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        dict_ = self.env.action_space.get_back_to_ref_state(obs)
        assert len(dict_) == 1
        assert "storage" in dict_
        assert np.all(
            np.abs(dict_["storage"][0].storage_p - 2.0 * self.env.storage_loss) <= 1e-5
        )  # I am in the original topology (up to the storage losses)

    def test_curtailment(self):
        obs, reward, done, info = self.env.step(
            self.env.action_space({"curtail": [(3, 0.05)]})
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "curtailment" in res
        assert len(res["curtailment"]) == 1
        for act in res["curtailment"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert np.all(obs.curtailment_limit == 1.0)
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology

        obs, reward, done, info = self.env.step(
            self.env.action_space({"curtail": [(3, 0.05)]})
        )
        obs, reward, done, info = self.env.step(
            self.env.action_space({"curtail": [(4, 0.5)]})
        )
        assert not done
        res = self.env.action_space.get_back_to_ref_state(obs)
        assert len(res) == 1
        assert "curtailment" in res
        assert len(res["curtailment"]) == 1
        for act in res["curtailment"]:
            obs, reward, done, info = self.env.step(act)
            assert not done
        assert np.all(obs.curtailment_limit == 1.0)
        assert (
            len(self.env.action_space.get_back_to_ref_state(obs)) == 0
        )  # I am in the original topology


# TODO test when not all action types are enable (typically the change / set part)
if __name__ == "__main__":
    unittest.main()
