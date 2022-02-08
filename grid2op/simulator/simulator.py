# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
from typing import Optional
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Exceptions import SimulatorError

class Simulator(object):
    def __init__(self, backend: Backend):
        # backend should be initiliazed !
        self.backend : Backend = backend.copy()
        self.current_obs : BaseObservation = None
        self._converged : Optional[bool] = None
        self._error : Optional[Exception] = None
    
    def copy(self) -> "Simulator":
        res = copy.copy(self)
        res.backend = res.backend.copy()
        if self.current_obs is not None:
            res.current_obs = res.current_obs.copy()
        return res
    
    def change_backend(self, backend):
        if not isinstance(backend, Backend):
            raise SimulatorError("when using change_backend function, the backend should"
                                 " be an object (an not a class) of type backend")
        self.backend.close()
        self.backend = backend.copy()  # backend_class.init_grid(type(self.backend))
        self.set_state(obs=self.current_obs)
    
    def change_backend_type(self, backend_type, load_grid_kwargs, **kwargs):
        if not isinstance(backend_type, type):
            raise SimulatorError("when using change_backend_type function, the backend_type should"
                                 " be a class an not an object")
        if not issubclass(backend_type, Backend):
            raise SimulatorError("when using change_backend_type function, the backend_type should"
                                 " be subtype of class Backend")
        tmp_backend = backend_type(**kwargs)
        tmp_backend.load_grid(**load_grid_kwargs)
        tmp_backend.assert_grid_correct()
        self.backend.close()
        self.backend = tmp_backend
        self.set_state(obs=self.current_obs)
        
    def set_state(self,
                  obs: Optional[BaseObservation]=None,
                  do_powerflow: bool=True,
                  new_gen_p : np.array=None,
                  new_gen_v : np.array=None,
                  new_load_p : np.array=None,
                  new_load_q : np.array=None):
        
        if obs is not None:
            self.current_obs = obs.copy()
            
        if self.current_obs is None:
            raise SimulatorError("The simulator is not initialized. Have you used `simulator.set_state(...)` with a valid observation before ?")
      
        # you cannot use "simulate" of the observation in this class
        self.current_obs._obs_env = None  
        self.current_obs._forecasted_inj = []
        self.current_obs._forecasted_grid = []
        
        # udpate the new state if needed
        if new_load_p is not None:
            self.current_obs.load_p[:] = new_load_p
        if new_load_q is not None:
            self.current_obs.load_q[:] = new_load_q
            
        if new_gen_p is not None:
            self.current_obs.gen_p[:] = new_gen_p
        if new_gen_v is not None:
            self.current_obs.gen_v[:] = new_gen_v
            
        self.converged = None
        self.error = None
        self.backend.update_from_obs(self.current_obs, force_update=True)
        
        if do_powerflow:
            self._do_powerflow()
    
    def _do_powerflow(self):
        self._converged, self._error = self.backend.runpf()
            
    def _update_obs(self):
        if self._converged:
            self.current_obs._update_attr_backend(self.backend)
        else:
            self.current_obs.set_game_over()
        
    def predict(self,
                act: BaseAction,
                do_copy: bool=True,
                new_gen_p : np.array=None,
                new_gen_v : np.array=None,
                new_load_p : np.array=None,
                new_load_q : np.array=None) -> "Simulator":
        # init the result
        if do_copy:
            res = self.copy()
        else:
            res = self
            
        res.set_state(obs=None, 
                      new_gen_p=new_gen_p,
                      new_gen_v=new_gen_v, 
                      new_load_p=new_load_p, 
                      new_load_q=new_load_q)
        
        # apply the action
        bk_act = res.backend.my_bk_act_class()
        bk_act += act
        res.backend.apply_action(bk_act)
        
        # run the powerflow
        res._do_powerflow()
        
        # update its observation
        res._update_obs()
        return res
