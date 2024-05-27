# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile
import numpy as np
import warnings
import unittest
try:
    from lightsim2grid import LightSimBackend
    LS_AVAIL = True
except ImportError:
    LS_AVAIL = False
    
import grid2op
from grid2op.Environment import TimedOutEnvironment, MaskedEnvironment, SingleEnvMultiProcess
from grid2op.Backend import PandaPowerBackend
from grid2op.Backend.educPandaPowerBackend import EducPandaPowerBackend
from grid2op.Episode import EpisodeData
from grid2op.Opponent import FromEpisodeDataOpponent
from grid2op.Runner import Runner
from grid2op.Action import TopologyAction, DispatchAction
from grid2op.tests.helper_path_test import *
from grid2op.Chronics import (FromHandlers,
                              Multifolder,
                              MultifolderWithCache,
                              GridStateFromFileWithForecasts,
                              GridStateFromFile,
                              GridStateFromFileWithForecastsWithMaintenance,
                              GridStateFromFileWithForecastsWithoutMaintenance,
                              FromOneEpisodeData,
                              FromMultiEpisodeData,
                              FromNPY)
from grid2op.Chronics.handlers import CSVHandler, JSONInitStateHandler


# TODO test "change" is deactivated
# TODO test with "names_orig_to_backend"


class TestSetActOrigDefault(unittest.TestCase):
    def _get_act_cls(self):
        return TopologyAction
    
    def _get_ch_cls(self):
        return Multifolder
    
    def _get_c_cls(self):
        return GridStateFromFileWithForecasts
    
    def _env_path(self):
        return os.path.join(
            PATH_DATA_TEST, "5bus_example_act_topo_set_init"
        )
    
    def _names_ch_to_bk(self):
        return None
    
    def _get_backend(self):
        return PandaPowerBackend()
    
    def _get_gridpath(self):
        return None
    
    def _get_envparams(self, env):
        return None
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        tmp_path = self._get_gridpath()
        env_params = dict(test=True,
                          backend=self._get_backend(),
                          action_class=self._get_act_cls(),
                          chronics_class=self._get_ch_cls(),
                          data_feeding_kwargs={"gridvalueClass": self._get_c_cls()},
                          _add_to_name=type(self).__name__
                         )
        if tmp_path is not None:
            env_params["grid_path"] = tmp_path
        ch_to_bk = self._names_ch_to_bk()
        if ch_to_bk is not None:
            env_params["names_chronics_to_grid"] = ch_to_bk
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm, **env_params)
        env_params = self._get_envparams(self.env)
        if env_params is not None:
            self.env.change_parameters(env_params)
            self.env.change_forecast_parameters(env_params)
        if issubclass(self._get_ch_cls(), MultifolderWithCache):
            self.env.chronics_handler.set_filter(lambda x: True)
            self.env.chronics_handler.reset()
        self.env.reset(seed=0)
        # some test to make sure the tests are correct
        assert issubclass(self.env.action_space.subtype, self._get_act_cls())
        assert isinstance(self.env.chronics_handler.real_data, self._get_ch_cls())
        assert isinstance(self.env.chronics_handler.real_data.data, self._get_c_cls())
        assert isinstance(self.env.backend, type(self._get_backend()))
        
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_reset_env(self, seed, ep_id):
        obs = self.env.reset(seed=seed, options={"time serie id": ep_id})
        return obs
    
    def _aux_make_step(self, act=None):
        if act is None:
            act = self.env.action_space()
        return self.env.step(act)
        
    def _aux_get_init_act(self):
        return self.env.chronics_handler.get_init_action(names_chronics_to_backend=self._names_ch_to_bk())
    
    def _aux_get_act_valid(self):
        # check the action in the time series folder is valid   
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            act_init = self._aux_get_init_act()
            
    def test_working_setbus(self):
        # ts id 0 => set_bus
        self.obs = self._aux_reset_env(seed=0, ep_id=0)
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        
        obs, reward, done, info = self._aux_make_step()
        assert not done
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 2
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        # check the action in the time series folder is valid
        self._aux_get_act_valid()
        
    def test_working_setstatus(self):
        # ts id 1 => set_status
        self.obs = self._aux_reset_env(seed=0, ep_id=1)
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not self.obs.line_status[1]
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        
        obs, reward, done, info = self._aux_make_step()
        assert not done
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == -1
        assert obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == -1
        assert not obs.line_status[1]
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        # check the action in the time series folder is valid
        self._aux_get_act_valid()
        
    def test_rules_ok(self):
        """test that even if the action to set is illegal, it works (case of ts id 2)"""
        self.obs = self._aux_reset_env(seed=0, ep_id=2)
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 2
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        act_init = self._aux_get_init_act()
        if act_init is None:
            # test not correct for multiprocessing, I stop here
            return
        obs, reward, done, info = self._aux_make_step(act_init)
        assert info["exception"] is not None
        assert info["is_illegal"]
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 2
        assert obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 2
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        # check the action in the time series folder is valid
        self._aux_get_act_valid()

    def test_change_bus_ignored(self, catch_warning=True):
        """test that if the action to set uses change_bus then nothing is done"""
        if catch_warning:
            with self.assertWarns(UserWarning):
                # it raises the warning "be carefull, change stuff are ignored"
                self.obs = self._aux_reset_env(seed=0, ep_id=3)
        else:
            # no warning in the main process in multiprocessing
            self.obs = self._aux_reset_env(seed=0, ep_id=3)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[2]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[2]] == 1
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 1
        assert self.obs.line_status[1] == 1
        assert self.obs.line_status[2] == 1
        assert self.obs.line_status[5] == 1
        # check the action in the time series folder is valid   
        self._aux_get_act_valid()
        

class TestSetActOrigDifferentActionCLS(TestSetActOrigDefault):
    def _get_act_cls(self):
        return DispatchAction


class TestSetAcOrigtMultiFolderWithCache(TestSetActOrigDefault):
    def _get_ch_cls(self):
        return MultifolderWithCache
    
    def test_two_reset_same(self):
        """test it does not crash when the same time series is used twice"""
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
    

class TestSetActOrigGridStateFromFile(TestSetActOrigDefault):
    def _get_c_cls(self):
        return GridStateFromFile
    
    
class TestSetActOrigGSFFWFWM(TestSetActOrigDefault):
    def _get_c_cls(self):
        return GridStateFromFileWithForecastsWithMaintenance
    
    
class TestSetActOrigGSFFWFWoM(TestSetActOrigDefault):
    def _get_c_cls(self):
        return GridStateFromFileWithForecastsWithoutMaintenance
    
    
class TestSetActOrigFromOneEpisodeData(TestSetActOrigDefault):
    def _aux_make_ep_data(self, ep_id):
        runner = Runner(**self.env.get_params_for_runner())
        runner.run(nb_episode=1,
                   episode_id=[ep_id],
                   path_save=self.fn.name,
                   max_iter=10)
        self.env.close()
        
        li_episode = EpisodeData.list_episode(self.fn.name)
        ep_data = li_episode[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    chronics_class=FromOneEpisodeData,
                                    data_feeding_kwargs={"ep_data": ep_data},
                                    opponent_class=FromEpisodeDataOpponent,
                                    opponent_attack_cooldown=1,
                                    _add_to_name=type(self).__name__
                                    )
        
    def setUp(self) -> None:
        self.fn = tempfile.TemporaryDirectory()
        super().setUp()
        
    def tearDown(self) -> None:
        self.fn.cleanup()
        return super().tearDown()
    
    def test_working_setbus(self):
        self._aux_make_ep_data(0)  # episode id 0 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setbus()
        
    def test_working_setstatus(self):
        self._aux_make_ep_data(1)  # episode id 1 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setstatus()
    
    def test_rules_ok(self):
        self._aux_make_ep_data(2)  # episode id 2 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_rules_ok()
                        
    def test_change_bus_ignored(self):
        self.skipTest("This make no sense for this class (change is not used internally)")


class TestSetActOrigFromMultiEpisodeData(TestSetActOrigDefault):
    def setUp(self) -> None:
        super().setUp()
        self.fn = tempfile.TemporaryDirectory()
        runner = Runner(**self.env.get_params_for_runner())
        runner.run(nb_episode=3,
                   episode_id=[0, 1, 2],
                   path_save=self.fn.name,
                   max_iter=10)
        self.env.close()
        
        li_episode = EpisodeData.list_episode(self.fn.name)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    chronics_class=FromMultiEpisodeData,
                                    data_feeding_kwargs={"li_ep_data": li_episode},
                                    opponent_class=FromEpisodeDataOpponent,
                                    opponent_attack_cooldown=1,
                                    _add_to_name=type(self).__name__
                                    )
        
        
    def tearDown(self) -> None:
        self.fn.cleanup()
        return super().tearDown()
    
    def test_two_reset_same(self):
        """test it does not crash when the same time series is used twice"""
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
        self.test_working_setstatus()
        obs, reward, done, info = self.env.step(self.env.action_space())
        
    def test_change_bus_ignored(self):
        self.skipTest("This make no sense for this class (change is not used internally)")
        
        
class TestSetActOrigFromNPY(TestSetActOrigDefault):
    def _aux_make_env(self, ch_id):
        self.obs = self.env.reset(seed=0, options={"time serie id": ch_id})
        load_p = 1.0 * self.env.chronics_handler._real_data.data.load_p[:self.max_iter,:]
        load_q = 1.0 * self.env.chronics_handler._real_data.data.load_q[:self.max_iter,:]
        gen_p = 1.0 * self.env.chronics_handler._real_data.data.prod_p[:self.max_iter,:]
        gen_v = np.repeat(self.obs.gen_v.reshape(1, -1), self.max_iter, axis=0)
        act = self.env.action_space({"set_bus": self.obs.topo_vect})
        self.env.close()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    chronics_class=FromNPY,
                                    data_feeding_kwargs={"load_p": load_p,
                                                         "load_q": load_q,
                                                         "prod_p": gen_p,
                                                         "prod_v": gen_v,
                                                         "init_state": act
                                                         },
                                    _add_to_name=type(self).__name__)
    def setUp(self) -> None:
        self.max_iter = 5
        super().setUp()
        
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_working_setbus(self):
        self._aux_make_env(0)  # episode id 0 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setbus()
        
    def test_working_setstatus(self):
        self._aux_make_env(1)  # episode id 1 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_working_setstatus()
    
    def test_rules_ok(self):
        self._aux_make_env(2)  # episode id 2 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_rules_ok()
    
    def test_change_bus_ignored(self):
        self._aux_make_env(3)  # episode id 3 is used for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().test_change_bus_ignored()
    

class TestSetActOrigEnvCopy(TestSetActOrigDefault):
    def setUp(self) -> None:
        super().setUp()
        env_cpy = self.env.copy()
        self.env.close()
        self.env = env_cpy


class TestSetActOrigFromHandlers(TestSetActOrigDefault):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self._env_path(),
                                    data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                         "gen_p_handler": CSVHandler("prod_p"),
                                                         "load_p_handler": CSVHandler("load_p"),
                                                         "gen_v_handler": CSVHandler("prod_v"),
                                                         "load_q_handler": CSVHandler("load_q"),
                                                         "init_state_handler": JSONInitStateHandler("init_state_handler")
                                                        },
                                    _add_to_name=type(self).__name__
                                    )


class TestSetActOrigLightsim(TestSetActOrigDefault):
    def _get_backend(self):
        if not LS_AVAIL:
            self.skipTest("LightSimBackend is not available")
        return LightSimBackend()


class TestSetActOrigDiffNames(TestSetActOrigDefault):
    def _get_gridpath(self):
        # just to have a grid with different names
        return os.path.join(PATH_DATA_TEST, "5bus_example_diff_name", "grid.json")
    
    def _names_ch_to_bk(self):
        res = {"loads": {'load_0_0': 'tutu', 'load_3_1': 'toto', 'load_4_2': 'tata'},
               "prods": {"gen_0_0": "othername_0_0", "gen_1_1": "othername_1_1"},
               "lines": {"0_1_0": 'l_0_1_0',
                         "0_2_1": 'l_0_2_1',
                         "0_3_2": 'l_0_3_2',
                         "0_4_3": 'l_0_4_3',
                         "1_2_4": 'l_1_2_4', 
                         "2_3_5": 'l_2_3_5',
                         "2_3_6": 'l_2_3_6',
                         "3_4_7": 'l_3_4_7'}
                }
        return res
    
    
class TestSetActOrigTOEnv(TestSetActOrigDefault):
    def setUp(self) -> None:
        super().setUp()
        env_init = self.env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = TimedOutEnvironment(self.env)
        env_init.close()
        return LightSimBackend()
    
    
class TestSetActOrigMaskedEnv(TestSetActOrigDefault):
    def setUp(self) -> None:
        super().setUp()
        env_init = self.env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = MaskedEnvironment(self.env,
                                         lines_of_interest=np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        env_init.close()
    

def always_true(x):
    # I can't use lambda in set_filter (lambda cannot be pickled)
    return True


class TestSetActOrigMultiProcEnv(TestSetActOrigDefault):
    def _aux_reset_env(self, seed, ep_id):
        # self.env.seed(seed)
        self.env.set_id(ep_id)
        obs = self.env.reset()
        return obs[0]

    def _aux_get_init_act(self):
        return None
    
    def _aux_make_step(self):
        obs, reward, done, info = self.env.step([self.env_init.action_space(), self.env_init.action_space()])
        return obs[0], reward[0], done[0], info[0]
            
    def setUp(self) -> None:
        super().setUp()
        self.env_init = self.env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = SingleEnvMultiProcess(self.env, 2)
        self.env.set_filter(always_true)
        
    def tearDown(self) -> None:
        self.env_init.close()
        return super().tearDown()
    
    def test_change_bus_ignored(self):
        super().test_change_bus_ignored(catch_warning=False)


class TestSetActOrigForcastEnv(TestSetActOrigDefault):
    def test_working_setbus(self):
        super().test_working_setbus()
        for_env = self.env.get_obs().get_forecast_env()
        obs, reward, done, info = for_env.step(self.env.action_space())
        
    def test_working_setstatus(self):
        super().test_working_setstatus()
        for_env = self.env.get_obs().get_forecast_env()
        obs, reward, done, info = for_env.step(self.env.action_space())
    
    def test_rules_ok(self):
        super().test_rules_ok()
        for_env = self.env.get_obs().get_forecast_env()
        obs, reward, done, info = for_env.step(self.env.action_space())
    
    def test_change_bus_ignored(self):
        super().test_change_bus_ignored()
        for_env = self.env.get_obs().get_forecast_env()
        obs, reward, done, info = for_env.step(self.env.action_space())


class TestSetActOrigRunner(unittest.TestCase):
    def _env_path(self):
        return TestSetActOrigDefault._env_path(self)
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True,
                                    _add_to_name=type(self).__name__
                                    )
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_right_init_act(self):
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=3,
                         episode_id=[0, 1, 2],
                         max_iter=10,
                         add_detailed_output=True)
        for i, el in enumerate(res):
            ep_data = el[-1]
            init_obs = ep_data.observations[0]
            if i == 0:
                assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 2
                assert init_obs.topo_vect[init_obs.load_pos_topo_vect[0]] == 2
                assert (init_obs.time_before_cooldown_line == 0).all()
                assert (init_obs.time_before_cooldown_sub == 0).all()
            elif i == 1:
                assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == -1
                assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[1]] == -1
                assert not init_obs.line_status[1]
                assert (init_obs.time_before_cooldown_line == 0).all()
                assert (init_obs.time_before_cooldown_sub == 0).all()
            elif i == 2:
                assert init_obs.topo_vect[init_obs.line_or_pos_topo_vect[1]] == 2
                assert init_obs.topo_vect[init_obs.line_ex_pos_topo_vect[5]] == 2
                assert (init_obs.time_before_cooldown_line == 0).all()
                assert (init_obs.time_before_cooldown_sub == 0).all()
            else:
                raise RuntimeError("Test is coded correctly")
                
                
class _PPNoShunt_Test(PandaPowerBackend):
    shunts_data_available = False
        
    
class TestSetSuntState(unittest.TestCase):
    def _env_path(self):
        return os.path.join(
            PATH_DATA_TEST, "educ_case14_storage_init_state"
        )
    
    def _get_backend(self):
        return PandaPowerBackend()
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True,
                                    _add_to_name=type(self).__name__
                                    )   
            self.env_noshunt = grid2op.make(self.env_nm,
                                            test=True,
                                            backend=_PPNoShunt_Test(),
                                            _add_to_name=type(self).__name__
                                            )   
            self.env_nostor = grid2op.make(self.env_nm,
                                           test=True,
                                           _add_to_name=type(self).__name__,
                                           _compat_glop_version="neurips_2020_compat"
                                           )   
        assert type(self.env_noshunt).shunts_data_available is False
        assert type(self.env_nostor).n_storage == 0
        assert type(self.env).n_storage == 2
        
    def test_set_shunt_state(self):
        """test that the action that acts on the shunt works (when shunt are supported)
        or has no impact if the backend does not support shunts"""
        obs_shunt = self.env.reset(seed=0, options={"time serie id": 0})
        obs_noshunt = self.env_noshunt.reset(seed=0, options={"time serie id": 0})
        assert obs_shunt._shunt_q[0] == 0.  # the action put the shunt to 0.
        # in the backend with no shunt, the shunt is active and generator 
        # does not produce same q
        assert abs(obs_shunt.gen_q[4] - obs_noshunt.gen_q[4]) > 5.
        
    def test_set_storage_state(self):
        obs_stor = self.env.reset(seed=0, options={"time serie id": 1})
        obs_nostor = self.env_nostor.reset(seed=0, options={"time serie id": 1})
        slack_id = -1
        # the storage action is taken into account
        assert obs_stor.storage_power[0] == 5.  # the action set this
        
        # the original grid (withtout storage)
        # and the grid with storage action have the same "gen_p"
        # if I remove the impact of the storage unit
        deltagen_p_th =  ((obs_stor.gen_p - obs_stor.actual_dispatch) - obs_nostor.gen_p)
        assert (np.abs(deltagen_p_th[:slack_id]) <= 1e-6).all()


class TestSetActOrigIgnoredParams(TestSetActOrigDefault):
    """This class test that the new feature (setting the initial state in the time series
    is properly ignored if the parameter says so)"""
    
    def _get_envparams(self, env):
        param = env.parameters
        param.IGNORE_INITIAL_STATE_TIME_SERIE = True
        return param
    
    def test_working_setbus(self):
        """test that it's ignored even if the action is set_status"""
        self.obs = self._aux_reset_env(seed=0, ep_id=0)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        
        obs, reward, done, info = self._aux_make_step()
        assert not done
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert obs.topo_vect[self.obs.load_pos_topo_vect[0]] == 1
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        # check the action in the time series folder is valid
        self._aux_get_act_valid()
        
    def test_working_setstatus(self):
        """test that it's ignored even if the action is set_status"""
        self.obs = self._aux_reset_env(seed=0, ep_id=1)
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.line_status[1]
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        
        obs, reward, done, info = self._aux_make_step()
        assert not done
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert obs.line_status[1]
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        # check the action in the time series folder is valid
        self._aux_get_act_valid()
        
    def test_rules_ok(self):
        """that it's ignored even if the action is illegal"""
        self.obs = self._aux_reset_env(seed=0, ep_id=2)
        
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 1
        assert (self.obs.time_before_cooldown_line == 0).all()
        assert (self.obs.time_before_cooldown_sub == 0).all()
        act_init = self._aux_get_init_act()
        if act_init is None:
            # test not correct for multiprocessing, I stop here
            return
        obs, reward, done, info = self._aux_make_step(act_init)
        assert info["exception"] is not None
        assert info["is_illegal"]
        assert obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 1
        assert (obs.time_before_cooldown_line == 0).all()
        assert (obs.time_before_cooldown_sub == 0).all()
        # check the action in the time series folder is valid
        self._aux_get_act_valid()

    def test_change_bus_ignored(self, catch_warning=True):
        """test that if the action to set uses change_bus then nothing is done"""
        # no warning in the main process in multiprocessing
        self.obs = self._aux_reset_env(seed=0, ep_id=3)
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] == 1
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[2]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[2]] == 1
        assert self.obs.topo_vect[self.obs.line_or_pos_topo_vect[5]] == 1
        assert self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[5]] == 1
        assert self.obs.line_status[1] == 1
        assert self.obs.line_status[2] == 1
        assert self.obs.line_status[5] == 1
        # check the action in the time series folder is valid   
        self._aux_get_act_valid()    
        
        
if __name__ == "__main__":
    unittest.main()
