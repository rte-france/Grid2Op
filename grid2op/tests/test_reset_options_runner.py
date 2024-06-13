# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import warnings
import unittest
    
import grid2op
from grid2op.Runner import Runner
from grid2op.tests.helper_path_test import *


class TestResetOptionRunner(unittest.TestCase):
    def _env_path(self):
        return "l2rpn_case14_sandbox"
    
    def setUp(self) -> None:
        self.env_nm = self._env_path()
        self.max_iter = 5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True
                                    )
        self.runner = Runner(**self.env.get_params_for_runner())
        
    def tearDown(self) -> None:
        self.env.close()
        self.runner._clean_up()
        return super().tearDown()
    
    def test_run_one_episode_ts_id(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # check it does not raise any error
            res = self.runner.run_one_episode(reset_options={"time serie id": 1},
                                              max_iter=self.max_iter,
                                              detailed_output=True
                                              )
        assert res[1]== '0001'
        assert res[3] == self.max_iter
        assert res[4] == self.max_iter
        
        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # check it does not raise any error
            res = self.runner.run_one_episode(reset_options={},
                                              episode_id=1,
                                              max_iter=self.max_iter,
                                              detailed_output=True
                                              )
        assert res[1]== '0001'
        assert res[3] == self.max_iter
        assert res[4] == self.max_iter
        
        # check the correct episode id is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run_one_episode(reset_options={"time serie id": 1},
                                              max_iter=self.max_iter,
                                              episode_id=0,
                                              detailed_output=True
                                              )
        assert res[1]== '0000'
        assert res[3] == self.max_iter
        assert res[4] == self.max_iter
        
    def test_run_one_episode_warning_raised_ts_id(self):
        # check it does raise an error
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")  
                res = self.runner.run_one_episode(reset_options={"time serie id": 1},
                                                max_iter=self.max_iter,
                                                episode_id=3,
                                                detailed_output=True
                                                )
        
    def test_run_onesingle_ep_ts_id(self):
        # one reset option
        res = self.runner.run(nb_episode=1,
                              reset_options={"time serie id": 1},
                               max_iter=self.max_iter
                               )
        assert res[0][1]== '0001'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # one list (of one element here)
        res = self.runner.run(nb_episode=1,
                              reset_options=[{"time serie id": 1}],
                              max_iter=self.max_iter
                              )
        assert res[0][1]== '0001'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # one tuple (of one element here)
        res = self.runner.run(nb_episode=1,
                              reset_options=({"time serie id": 1}, ),
                              max_iter=self.max_iter
                              )
        assert res[0][1]== '0001'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # check the correct episode id is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(nb_episode=1,
                                  reset_options={"time serie id": 1},
                                  max_iter=self.max_iter,
                                  episode_id=[0]
                                  )
        assert res[0][1]== '0000'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # check the warning is raised
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                res = self.runner.run(nb_episode=1,
                                      reset_options={"time serie id": 1},
                                      max_iter=self.max_iter,
                                      episode_id=[0]
                                      )
        
    def test_run_two_eps_seq_ts_id(self, nb_process=1):
        # one reset option
        res = self.runner.run(nb_episode=2,
                              reset_options={"time serie id": 1},
                               max_iter=self.max_iter,
                               nb_process=nb_process
                               )
        for el in res:
            assert el[1]== '0001'
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # one list (of one element here)
        res = self.runner.run(nb_episode=2,
                              reset_options=[{"time serie id": 1}, {"time serie id": 1}],
                              max_iter=self.max_iter,
                               nb_process=nb_process
                              )
        for el in res:
            assert el[1]== '0001'
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # one tuple (of one element here)
        res = self.runner.run(nb_episode=2,
                              reset_options=({"time serie id": 1}, {"time serie id": 1}),
                              max_iter=self.max_iter,
                               nb_process=nb_process
                              )
        for el in res:
            assert el[1]== '0001'
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the correct episode id is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(nb_episode=2,
                                  reset_options={"time serie id": 1},
                                  max_iter=self.max_iter,
                                  episode_id=[0, 1],
                                  nb_process=nb_process
                                  )
        assert res[0][1]== '0000'
        assert res[1][1]== '0001'
        for el in res:
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the warning is raised
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                res = self.runner.run(nb_episode=2,
                                      reset_options={"time serie id": 1},
                                      max_iter=self.max_iter,
                                      episode_id=[0, 1],
                                      nb_process=nb_process
                                     )
        
    def test_run_two_eps_seq_two_options_ts_id(self, nb_process=1):          
        # one list (of one element here)
        res = self.runner.run(nb_episode=2,
                              reset_options=[{"time serie id": 0}, {"time serie id": 1}],
                              max_iter=self.max_iter,
                               nb_process=nb_process
                              )
        assert res[0][1]== '0000'
        assert res[1][1]== '0001'
        for el in res:
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # one tuple (of one element here)
        res = self.runner.run(nb_episode=2,
                              reset_options=({"time serie id": 0}, {"time serie id": 1}),
                              max_iter=self.max_iter,
                               nb_process=nb_process
                              )
        assert res[0][1]== '0000'
        assert res[1][1]== '0001'
        for el in res:
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the correct episode id is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(nb_episode=2,
                                  reset_options={"time serie id": 1},
                                  max_iter=self.max_iter,
                                  episode_id=[0, 1],
                                  nb_process=nb_process
                                  )
        assert res[0][1]== '0000'
        assert res[1][1]== '0001'
        for el in res:
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the warning is raised
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                res = self.runner.run(nb_episode=2,
                                      reset_options={"time serie id": 1},
                                      max_iter=self.max_iter,
                                      episode_id=[0, 1],
                                      nb_process=nb_process
                                     )
        
    def test_run_two_eps_par_ts_id(self):
        self.test_run_two_eps_seq_ts_id(nb_process=2)
        
    def test_run_two_eps_par_two_opts_ts_id(self):
        self.test_run_two_eps_seq_two_options_ts_id(nb_process=2)
    
    def test_fail_when_needed(self):
        # wrong type
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                  reset_options=1,
                                  episode_id=[0, 1],
                                  max_iter=self.max_iter,
                                  add_detailed_output=True,
                                  )
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                reset_options=[1, {"time serie id": 1}],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                reset_options=[{"time serie id": 1}, 1],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
            
        # wrong size (too big)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                reset_options=[{"time serie id": 1},
                                               {"time serie id": 1},
                                               {"time serie id": 1}],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
        # wrong size (too small)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                reset_options=[{"time serie id": 1}],
                                episode_id=[0, 1],
                                max_iter=self.max_iter,
                                add_detailed_output=True,
                                )
        # wrong key (beginning)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                 reset_options=[{"bleurk": 1}, {"time serie id": 1}],
                                 episode_id=[0, 1],
                                 max_iter=self.max_iter,
                                 add_detailed_output=True,
                                 )
        # wrong key (end)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                 reset_options=[{"time serie id": 1}, {"bleurk": 1}],
                                 episode_id=[0, 1],
                                 max_iter=self.max_iter,
                                 add_detailed_output=True,
                                 )
        # wrong key (when alone)
        with self.assertRaises(RuntimeError):
            res = self.runner.run(nb_episode=2,
                                    reset_options={"bleurk": 1},
                                    episode_id=[0, 1],
                                    max_iter=self.max_iter,
                                    add_detailed_output=True,
                                    )
    
    def test_run_one_episode_max_it(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # check it does not raise any error
            res = self.runner.run_one_episode(reset_options={"max step": self.max_iter, "time serie id": 1},
                                              detailed_output=True
                                              )
        assert res[1]== '0001'
        assert res[3] == self.max_iter
        assert res[4] == self.max_iter
        
        with warnings.catch_warnings():
            warnings.filterwarnings("error")  # check it does not raise any error
            res = self.runner.run_one_episode(reset_options={"time serie id": 1},
                                              max_iter=self.max_iter,
                                              detailed_output=True
                                              )
        assert res[1]== '0001'
        assert res[3] == self.max_iter
        assert res[4] == self.max_iter
        
        # check the correct max iter is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run_one_episode(reset_options={"time serie id": 1, "max step": self.max_iter + 1},
                                              max_iter=self.max_iter,
                                              episode_id=0,
                                              detailed_output=True
                                             )
        assert res[1]== '0000'
        assert res[3] == self.max_iter
        assert res[4] == self.max_iter
        
    def test_run_one_episode_warning_raised_max_it(self):
        # check it does raise an error
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")  
                res = self.runner.run_one_episode(reset_options={"time serie id": 1, "max step": self.max_iter + 3},
                                                  max_iter=self.max_iter
                                                 )
        
    def test_run_onesingle_ep_max_it(self):
        # one reset option
        res = self.runner.run(nb_episode=1,
                              reset_options={"time serie id": 1, "max step": self.max_iter},
                               )
        assert res[0][1]== '0001'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # one list (of one element here)
        res = self.runner.run(nb_episode=1,
                              reset_options=[{"time serie id": 1, "max step": self.max_iter}],
                              )
        assert res[0][1]== '0001'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # one tuple (of one element here)
        res = self.runner.run(nb_episode=1,
                              reset_options=({"time serie id": 1, "max step": self.max_iter}, ),
                              )
        assert res[0][1]== '0001'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # check the correct episode id is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(nb_episode=1,
                                  reset_options={"time serie id": 0, "max step": self.max_iter + 3},
                                  max_iter=self.max_iter,
                                  )
        assert res[0][1]== '0000'
        assert res[0][3] == self.max_iter
        assert res[0][4] == self.max_iter
        
        # check the warning is raised
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                res = self.runner.run(nb_episode=1,
                                      reset_options={"time serie id": 0, "max step": self.max_iter + 3},
                                      max_iter=self.max_iter
                                      )
        
    def test_run_two_eps_seq_max_it(self, nb_process=1):
        # one reset option
        res = self.runner.run(nb_episode=2,
                              reset_options={"time serie id": 1, "max step": self.max_iter },
                               nb_process=nb_process
                               )
        for el in res:
            assert el[1]== '0001'
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # one list (of the same element here)
        res = self.runner.run(nb_episode=2,
                              reset_options=[{"time serie id": 1, "max step": self.max_iter},
                                             {"time serie id": 1, "max step": self.max_iter}],
                               nb_process=nb_process
                              )
        for el in res:
            assert el[1]== '0001'
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # one tuple (of the same element here)
        res = self.runner.run(nb_episode=2,
                              reset_options=({"time serie id": 1, "max step": self.max_iter},
                                             {"time serie id": 1, "max step": self.max_iter}),
                               nb_process=nb_process
                              )
        for el in res:
            assert el[1]== '0001'
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the correct "max iter" is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(nb_episode=2,
                                  reset_options={"max step": self.max_iter + 3},
                                  max_iter=self.max_iter,
                                  episode_id=[0, 1],
                                  nb_process=nb_process
                                  )
        assert res[0][1]== '0000'
        assert res[1][1]== '0001'
        for el in res:
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the warning is raised
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                res = self.runner.run(nb_episode=2,
                                      reset_options={"max step": self.max_iter + 3},
                                      max_iter=self.max_iter,
                                      episode_id=[0, 1],
                                      nb_process=nb_process
                                     )
        
    def test_run_two_eps_seq_two_options_max_it(self, nb_process=1):          
        # one list (of two different elements here)
        res = self.runner.run(nb_episode=2,
                              reset_options=[{"time serie id": 0, "max step": self.max_iter + 1},
                                             {"time serie id": 1, "max step": self.max_iter + 2}],
                               nb_process=nb_process
                              )
        assert res[0][1]== '0000'
        assert res[0][3] == self.max_iter + 1
        assert res[0][4] == self.max_iter + 1
        assert res[1][1]== '0001'
        assert res[1][3] == self.max_iter + 2
        assert res[1][4] == self.max_iter + 2
        
        # one tuple (of two different elements here)
        res = self.runner.run(nb_episode=2,
                              reset_options=({"time serie id": 0, "max step": self.max_iter + 1},
                                             {"time serie id": 1, "max step": self.max_iter + 2}),
                               nb_process=nb_process
                              )
        assert res[0][1]== '0000'
        assert res[0][3] == self.max_iter + 1
        assert res[0][4] == self.max_iter + 1
        assert res[1][1]== '0001'
        assert res[1][3] == self.max_iter + 2
        assert res[1][4] == self.max_iter + 2
        
        # check the correct max iter is used
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.runner.run(nb_episode=2,
                                  reset_options={"max step": self.max_iter + 1},
                                  max_iter=self.max_iter,
                                  episode_id=[0, 1],
                                  nb_process=nb_process
                                  )
        assert res[0][1]== '0000'
        assert res[1][1]== '0001'
        for el in res:
            assert el[3] == self.max_iter
            assert el[4] == self.max_iter
        
        # check the warning is raised
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                res = self.runner.run(nb_episode=2,
                                      reset_options={"max step": self.max_iter + 1},
                                      max_iter=self.max_iter,
                                      episode_id=[0, 1],
                                      nb_process=nb_process
                                     )
        
    def test_run_two_eps_par_max_it(self):
        self.test_run_two_eps_seq_max_it(nb_process=2)
        
    def test_run_two_eps_par_two_opts_max_it(self):
        self.test_run_two_eps_seq_two_options_max_it(nb_process=2)
        
        
    #####################
                    
        
if __name__ == "__main__":
    unittest.main()
