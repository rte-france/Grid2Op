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
from grid2op.Environment import BaseEnv
from grid2op.Observation import CompleteObservation
from grid2op.Parameters import Parameters
from grid2op.Reward import GameplayReward, L2RPNReward
from grid2op.Exceptions import EnvError, NoForecastAvailable
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import BaseOpponent
from grid2op.dtypes import dt_float

import warnings

warnings.simplefilter("error")


class TestMultiMixEnvironment(unittest.TestCase):
    def test_creation(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        assert mme.current_obs is not None
        assert mme.current_env is not None

    def test_get_path_env(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        path_mme = mme.get_path_env()
        for mix in mme:
            path_mix = mix.get_path_env()
            assert path_mme != path_mix
            assert os.path.split(path_mix)[0] == path_mme

    def test_create_fail(self):
        with self.assertRaises(EnvError):
            mme = MultiMixEnvironment("/tmp/error")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(EnvError):
                mme = MultiMixEnvironment(tmpdir)

    def test_creation_with_params(self):
        p = Parameters()
        p.MAX_SUB_CHANGED = 666
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, param=p, _test=True)
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
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, param=p, 
                                  other_rewards=oth_r, _test=True)
        assert mme.current_obs is not None
        assert mme.current_env is not None
        o, r, d, i = mme.step(mme.action_space({}))
        assert i is not None
        assert "rewards" in i
        assert "game" in i["rewards"]
        assert "l2rpn" in i["rewards"]

    def test_creation_with_backend(self):
        class DummyBackend1(PandaPowerBackend):
            def dummy(self):
                return True

        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, 
                                  backend=DummyBackend1(), 
                                  _test=True)
        assert mme.current_obs is not None
        assert mme.current_env is not None
        for env in mme:
            assert env.backend.dummy() == True

    def test_creation_with_backend_are_not_shared(self):
        class DummyBackend2(PandaPowerBackend):
            def __init__(self, detailed_infos_for_cascading_failures=False,
                         can_be_copied=True,
                         lightsim2grid=False,
                         dist_slack=False,
                         max_iter=10):
                super().__init__(
                    detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                    can_be_copied=can_be_copied,
                    lightsim2grid=lightsim2grid,
                    dist_slack=dist_slack,
                    max_iter=max_iter
                )
                self.calls = 0

            def dummy(self):
                r = self.calls
                self.calls += 1
                return r

        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                  backend=DummyBackend2(),
                                  _test=True)
        assert mme.current_obs is not None
        assert mme.current_env is not None
        for t in range(3):
            for env in mme:
                dummy = env.backend.dummy()
                assert dummy == t

    def test_creation_with_opponent(self):
        mme = MultiMixEnvironment(
            PATH_DATA_MULTIMIX,
            opponent_class=BaseOpponent,
            opponent_init_budget=42.0,
            opponent_budget_per_ts=0.42,
            _test=True
        )
        assert mme.current_obs is not None
        assert mme.current_env is not None
        for env in mme:
            assert env._opponent_class == BaseOpponent
            assert env._opponent_init_budget == dt_float(42.0)
            assert env._opponent_budget_per_ts == dt_float(0.42)

    def test_reset(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        mme.reset()
        assert mme.current_obs is not None
        assert mme.current_env is not None

    def test_reset_with_params(self):
        p = Parameters()
        p.MAX_SUB_CHANGED = 666
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, param=p, _test=True)
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
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, param=p,
                                  other_rewards=oth_r, _test=True)
        mme.reset()

        assert mme.current_obs is not None
        assert mme.current_env is not None
        o, r, d, i = mme.step(mme.action_space({}))
        assert i is not None
        assert "rewards" in i
        assert "game" in i["rewards"]
        assert "l2rpn" in i["rewards"]

    def test_reset_with_backend(self):
        class DummyBackend3(PandaPowerBackend):
            def __init__(self, 
                         detailed_infos_for_cascading_failures=False,
                         can_be_copied=True,
                         lightsim2grid=False,
                         dist_slack=False,
                         max_iter=10
                         ):
                super().__init__(
                    detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                    can_be_copied=can_be_copied,
                    lightsim2grid=lightsim2grid,
                    dist_slack=dist_slack,
                    max_iter=max_iter
                )
                self._dummy = -1

            def reset(self, grid_path=None, grid_filename=None):
                self._dummy = 1

            def dummy(self):
                return self._dummy

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            mme = MultiMixEnvironment(PATH_DATA_MULTIMIX,
                                    backend=DummyBackend3(),
                                    _test=True)
            mme.reset()
        assert mme.current_env.backend.dummy() == 1

    def test_reset_with_opponent(self):
        mme = MultiMixEnvironment(
            PATH_DATA_MULTIMIX,
            opponent_class=BaseOpponent,
            opponent_init_budget=42.0,
            opponent_budget_per_ts=0.42, 
            _test=True,
        )
        mme.reset()
        assert mme.current_obs is not None
        assert mme.current_env is not None
        assert mme._opponent_class == BaseOpponent
        assert mme._opponent_init_budget == dt_float(42.0)
        assert mme._opponent_budget_per_ts == dt_float(0.42)

    def test_reset_seq(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        for i in range(2):
            assert i == mme.current_index
            mme.reset()
            assert mme.current_obs is not None
            assert mme.current_env is not None

    def test_reset_random(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        for i in range(2):
            mme.reset(random=True)
            assert mme.current_obs is not None
            assert mme.current_env is not None

    def test_seeding(self):
        mme1 = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        seeds_1 = mme1.seed(2)
        mme1.close()

        mme2 = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        seeds_2 = mme2.seed(42)
        mme2.close()

        mme3 = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        seeds_3 = mme3.seed(2)
        mme3.close()

        assert np.all(seeds_1 == seeds_3)
        assert np.any(seeds_1 != seeds_2)

    def test_step_dn(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        dn = mme.action_space({})

        obs, r, done, info = mme.step(dn)
        assert obs is not None
        assert r is not None
        assert isinstance(info, dict)
        assert done is not True

    def test_step_switch_line(self):
        LINE_ID = 4
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        line_ex_topo = mme.line_ex_pos_topo_vect[LINE_ID]
        line_or_topo = mme.line_or_pos_topo_vect[LINE_ID]
        switch_status = mme.action_space.get_change_line_status_vect()
        switch_status[LINE_ID] = True
        switch_action = mme.action_space({"change_line_status": switch_status})

        obs, r, d, info = mme.step(switch_action)
        assert d is False
        assert obs.line_status[LINE_ID] == False
        obs, r, d, info = mme.step(switch_action)
        assert d is False, "Diverged powerflow on reconnection"
        assert info["is_illegal"] == False, "Reconnecting should be legal"
        assert obs.line_status[LINE_ID] == True, "Line is not reconnected"

    def test_forecast_toggle(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        dn = mme.action_space({})
        # Forecast off
        mme.deactivate_forecast()
        # Step once
        obs, _, _, _ = mme.step(dn)
        # Cant simulate
        with self.assertRaises(NoForecastAvailable):
            obs.simulate(dn)
        # Forecast ON
        mme.reactivate_forecast()
        # Reset, step once
        mme.reset()
        obs, _, _, _ = mme.step(dn)
        # Can simulate
        obs, r, done, info = obs.simulate(dn)
        assert obs is not None
        assert r is not None
        assert isinstance(info, dict)
        assert done is not True

    def test_bracket_access_by_name(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)

        mix1_env = mme["case14_001"]
        assert mix1_env.name == "case14_001"
        mix2_env = mme["case14_002"]
        assert mix2_env.name == "case14_002"
        with self.assertRaises(KeyError):
            unknown_env = mme["unknown_raise"]

    def test_keys_access(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)

        for k in mme.keys():
            mix = mme[k]
            assert mix is not None
            assert isinstance(mix, BaseEnv)
            assert mix.name == k

    def test_values_access(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)

        for v in mme.values():
            assert v is not None
            assert isinstance(v, BaseEnv)
            assert v == mme[v.name]

    def test_values_unique(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        vals = list(mme.values())
        vals_unique = list(set(vals))

        assert len(vals) == len(vals_unique)

    def test_items_acces(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)

        for k, v in mme.items():
            assert k is not None
            assert v is not None
            assert isinstance(v, BaseEnv)
            assert v == mme[k]

    def test_copy(self):
        # https://github.com/BDonnot/lightsim2grid/issues/10
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX, _test=True)
        for i in range(5):
            obs, reward, done, info = mme.step(mme.action_space())
        env2 = mme.copy()

        obsnew = env2.get_obs()
        assert obsnew == obs

        # after the same action, the original env and its copy are the same
        obs0, reward0, done0, info0 = mme.step(mme.action_space())
        obs1, reward1, done1, info1 = env2.step(env2.action_space())
        assert obs0 == obs1
        assert reward0 == reward1
        assert done1 == done0

        # reset has the correct behaviour
        obs_after = env2.reset()
        obs00, reward00, done00, info00 = mme.step(mme.action_space())
        # i did not affect the other environment
        assert (
            obs00.minute_of_hour
            == obs0.minute_of_hour + mme.chronics_handler.time_interval.seconds // 60
        )
        # reset read the right chronics
        assert obs_after.minute_of_hour == 0


if __name__ == "__main__":
    unittest.main()
