# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile

from grid2op.tests.helper_path_test import *
from grid2op.Environment import MultiMixEnvironment
from grid2op.Observation import CompleteObservation
from grid2op.Parameters import Parameters
from grid2op.Reward import GameplayReward, L2RPNReward
from grid2op.Exceptions import EnvError, NoForecastAvailable
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import BaseOpponent
from grid2op.dtypes import dt_float

class TestMultiMixEnvironment(unittest.TestCase):
    def test_creation(self):        
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        assert mme.current_obs is not None
        assert mme.current_env is not None

    def test_create_fail(self):
        with self.assertRaises(EnvError):
            mme = MultiMixEnvironment("/tmp/error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(EnvError):
                mme = MultiMixEnvironment(tmpdir)

    def test_creation_with_params(self):
        p = Parameters()
        p.MAX_SUB_CHANGED = 666
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, param=p)
        assert mme.current_obs is not None
        assert mme.current_env is not None
        assert mme.parameters.MAX_SUB_CHANGED == 666

    def test_creation_with_other_rewards(self):
        p = Parameters()
        p.NO_OVERFLOW_DISCONNECTION = True
        oth_r = {
            "game": GameplayReward,
            "l2rpn": L2RPNReward,
        }
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  param=p,
                                  other_rewards=oth_r)
        assert mme.current_obs is not None
        assert mme.current_env is not None
        o, r, d, i = mme.step(mme.action_space({}))
        assert i is not None
        assert "rewards" in i
        assert "game" in i["rewards"]
        assert "l2rpn" in i["rewards"]

    def test_creation_with_backend(self):
        class DummyBackend(PandaPowerBackend):
            def dummy(self):
                return True
            
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  backend=DummyBackend())
        assert mme.current_obs is not None
        assert mme.current_env is not None
        for env in mme._envs:
            assert env.backend.dummy() == True

    def test_creation_with_opponent(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  opponent_class=BaseOpponent,
                                  opponent_init_budget=42.0,
                                  opponent_budget_per_ts=0.42)
        assert mme.current_obs is not None
        assert mme.current_env is not None
        for env in mme._envs:
            assert env.opponent_class == BaseOpponent
            assert env.opponent_init_budget == dt_float(42.0)
            assert env.opponent_budget_per_ts == dt_float(0.42)

    def test_reset(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        mme.reset()
        assert mme.current_obs is not None
        assert mme.current_env is not None

    def test_reset_with_params(self):
        p = Parameters()
        p.MAX_SUB_CHANGED = 666
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, param=p)
        mme.reset()
        assert mme.current_obs is not None
        assert mme.current_env is not None
        assert mme.parameters.MAX_SUB_CHANGED == 666

    def test_reset_with_other_rewards(self):
        p = Parameters()
        p.NO_OVERFLOW_DISCONNECTION = True
        oth_r = {
            "game": GameplayReward,
            "l2rpn": L2RPNReward,
        }
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  param=p,
                                  other_rewards=oth_r)
        mme.reset()

        assert mme.current_obs is not None
        assert mme.current_env is not None
        o, r, d, i = mme.step(mme.action_space({}))
        assert i is not None
        assert "rewards" in i
        assert "game" in i["rewards"]
        assert "l2rpn" in i["rewards"]

    def test_reset_with_backend(self):
        class DummyBackend(PandaPowerBackend):
            self._dummy = -1

            def reset(self):
                self._dummy = 1
                
            def dummy(self):
                return self._dummy
            
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  backend=DummyBackend())
        mme.reset()
        for env in mme._envs:
            assert env.backend.dummy() == 1

    def test_reset_with_opponent(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  opponent_class=BaseOpponent,
                                  opponent_init_budget=42.0,
                                  opponent_budget_per_ts=0.42)
        mme.reset()
        assert mme.current_obs is not None
        assert mme.current_env is not None
        assert mme.opponent_class == BaseOpponent
        assert mme.opponent_init_budget == dt_float(42.0)
        assert mme.opponent_budget_per_ts == dt_float(0.42)

    def test_reset_seq(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        for i in range(2):
            assert i == mme.current_index
            mme.reset()
            assert mme.current_obs is not None
            assert mme.current_env is not None

    def test_reset_random(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        for i in range(2):
            mme.reset(random=True)
            assert mme.current_obs is not None
            assert mme.current_env is not None

        
    def test_seeding(self):
        mme1 = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        seeds_1 = mme1.seed(2)
        mme1.close()
        
        mme2 = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        seeds_2 = mme2.seed(42)
        mme2.close()
        
        mme3 = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        seeds_3 = mme3.seed(2)
        mme3.close()

        assert np.all(seeds_1 == seeds_3)
        assert np.any(seeds_1 != seeds_2)

    def test_step_dn(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        dn = mme.action_space({})

        obs, r, done, info  = mme.step(dn)
        assert obs is not None
        assert r is not None
        assert isinstance(info, dict)
        assert done is not True

    def test_step_switch_line(self):
        LINE_ID = 4
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        line_ex_topo = mme.line_ex_pos_topo_vect[LINE_ID]
        line_or_topo = mme.line_or_pos_topo_vect[LINE_ID]
        switch_status = mme.action_space.get_change_line_status_vect()
        switch_status[LINE_ID] = True
        switch_action = mme.action_space({
            'change_line_status': switch_status
        })

        obs, r, d, info = mme.step(switch_action)
        assert d is False
        assert obs.line_status[LINE_ID] == False
        obs, r, d, info = mme.step(switch_action)
        assert d is False, "Diverged powerflow on reconnection"
        assert info["is_illegal"] == False, "Reconnecting should be legal"
        assert obs.line_status[LINE_ID] == True, "Line is not reconnected"

    def test_forecast_toggle(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        dn = mme.action_space({})
        # Forecast off
        mme.deactivate_forecast()
        # Step once
        obs, _, _ , _ = mme.step(dn)
        # Cant simulate
        with self.assertRaises(NoForecastAvailable):
            obs.simulate(dn)
        # Forecast ON
        mme.reactivate_forecast()
        # Reset, step once
        mme.reset()
        obs, _, _ , _ = mme.step(dn)
        # Can simulate
        obs, r, done, info = obs.simulate(dn)
        assert obs is not None
        assert r is not None
        assert isinstance(info, dict)
        assert done is not True

if __name__ == "__main__":
    unittest.main()
