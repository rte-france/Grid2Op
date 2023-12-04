# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import os
import numpy as np
from typing import Optional, Tuple
from grid2op.Space import RandomObject
from datetime import timedelta, datetime


# TODO logger !
class BaseHandler(RandomObject):
    """This is the base class that represents a time series "handler".
    
    .. versionadded:: 1.9.0
    
    Each "handler" will be reponsible to produce the data for "one single type of elements" 
    of the grid. For example you will have 1 handler for "load_p", one for "load_q", another
    one for "load_p_forecasted" etc. This allows some great flexibility to the way you want 
    to retrieve data, but can be quite verbose as every type of data needs to be "handled".
    
    A "handler" will, for a certain type of data (*eg* load_p or maintenance etc.)
    handle the way this data type is generated.
    
    To be a valid "handler" an class must first inherit from :class:`BaseHandler` and 
    implements (for all types of handlers):
    
    - :func:`BaseHandler.initialize` : to initialize the handler from the environment 
      data (number of loads, lines etc.)
    - :func:`BaseHandler.done` : whether or not this "handler" is over or not.
    - :func:`BaseHandler.check_validity` : check if the input data are valid with the backend, 
      for example if you read from a csv
      number of columns should match number of element
    - :func:`BaseHandler.next_chronics` : called just before the start of a scenario.
    
    If the data represents "real time" data (*ie* the data seen by the agent in real 
    time in the observation) then it needs also to implement:
    
    - :func:`BaseHandler.load_next` : to "generate" data for the next steps
    
    If the data represents "forecast data" (*ie* the data accessed by the agent when it uses
    :func:`grid2op.Observation.BaseObservation.simulate` or :class:`grid2op.simulator.Simulator`
    or :func:`grid2op.Observation.BaseObservation.get_forecast_env`) then it needs to implement:
    
    - :func:`BaseHandler.forecast` : to retrieve the forecast at a given horizon
    
    And if the "handler" represents maintenance data, then it needs to implement:
    
    - :func:`BaseHandler.load_next` : that returns a boolean vector for whether or not 
      each powerline is in maintenance
    - :func:`BaseHandler.load_next_maintenance` : to "generate" data for the next steps
    
    .. seealso:: 
        
        :class:`grid2op.Chronics.FromHandlers` which is the things that "consumes" the handlers to output 
        the data read by the :class:`grid2op.Environment.Environment`
    
    """
    def __init__(self, array_name, max_iter=-1, h_forecast=(5, )):
        super().__init__()
        self.max_iter : int = max_iter
        self.init_datetime : Optional[datetime] = None
        self.time_interval : Optional[timedelta] = None
        self.array_name : str = array_name
        self._h_forecast : tuple = copy.deepcopy(h_forecast)
        self.path : Optional[os.PathLike] = None
        self.max_episode_duration : Optional[int] = None
    
    def set_max_iter(self, max_iter: Optional[int]) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        The the maximum number of iteration this handler is able to produce.
        
        `-1` means "forever". This is set by the :class:`grid2op.Chronics.FromHandlers`

        Parameters
        ----------
        max_iter : Optional[int]
            Maximum number of iterations
            
        """
        if max_iter is not None:
            self.max_iter = int(max_iter)
        else:
            self.max_iter = -1
            
    def set_max_episode_duration(self, max_episode_duration: Optional[int]) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        The the maximum number of iteration the environment will use. It is a way
        to synchronize all the handlers for the same environment

        Parameters
        ----------
        max_episode_duration : Optional[int]
            Maximum number of iterations for the current grid2op environment
            
        """
        if max_episode_duration is not None:
            self.max_episode_duration = int(max_episode_duration)
        else:
            self.max_episode_duration = None
        
    def get_max_iter(self) -> int:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        The maximum number of iterations this particular handler can generate.
        
        `-1` means "forever" otherwise it should be a > 0 integers.

        Returns
        -------
        int
            The maximum number of iterations this particular handler can generate.
        """
        return self.max_iter
    
    def set_path(self, path: os.PathLike) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This method is used by the :class:`grid2op.Chronics.FromHandlers` to inform this handler about the location
        where the required data for this handler could be located.

        Parameters
        ----------
        path : os.PathLike
            The path to look for the data
        """
        self.path = path
    
    def set_chunk_size(self, chunk_size: Optional[int]) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        optional: when data are read from the hard drive (*eg* :class:`grid2op.Chronics.handlers.CSVHandler`)
        this can inform the handler about the number of data to proceed at each 'step'.
        
        .. note::
           Do not use this function directly, it should be used only from the environment.
           
        .. seealso::
            This can be set by a call to `env.chronics_handler.set_chunk_size(chunk_size)`

        Parameters
        ----------
        chunk_size : Optional[int]
            The desired chunk size
        """
        # Chunk size is part of public API but has no sense for 
        # data not read from a disk
        pass
        
    def set_times(self,
                  init_datetime : datetime,
                  time_interval : timedelta) -> None:

        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This method is used by the :class:`grid2op.Chronics.FromHandlers` to inform this handler 
        about the intial datetime of the episode and the duration between two steps.
        
        .. note::
           Do not use this function directly, it should be used only from the environment.
           
        Parameters
        ----------
        init_datetime : datetime
            The initial datetime.
            
        time_interval : timedelta
            The time between two steps of the environment.
            
        """
        self.init_datetime = init_datetime
        self.time_interval = time_interval
    
    def _clear(self):
        self.init_datetime = None
        self.time_interval = None

    def get_kwargs(self, dict_ : dict) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        If some extra parameters are needed to build your handler "from scratch" you should copy them here. This is used
        when creating a runner for example.

        Parameters
        ----------
        dict_ : dict
            The dictionnary to update with the parameters.
        """
        # no need to remember special kwargs for the base class
        pass
    
    def set_h_forecast(self, h_forecast : Tuple[int]) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This method is used by the :class:`grid2op.Chronics.FromHandlers` to inform this handler 
        about the different forecast horizons available.

        .. seealso::
            :func:`BaseHandler.get_available_horizons`
            
        Parameters
        ----------
        h_forecast : Tuple[int]
            A tuple containing the different forecast horizons available. The horizons should be given in minutes,
            for example `handler.set_h_forecast((5, 10))` tells this handler that forecasts are available for 5 and 
            10 minutes ahead.
            
        """
        self._h_forecast = copy.deepcopy(h_forecast)
    
    def get_available_horizons(self) -> Tuple:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This methods returns the available forecast horizons (in minutes) known by this handler.
        
        .. seealso::
            :func:`BaseHandler.set_h_forecast`
            
        Returns
        -------
        Tuple
            A tuple containing the different forecast horizons available. The horizons should be given in minutes,
            for example `handler.set_h_forecast((5, 10))` tells this handler that forecasts are available for 5 and 
            10 minutes ahead.
            
        """
        return copy.deepcopy(self._h_forecast)
        
    def initialize(self,
                   order_backend_arrays,
                   names_chronics_to_backend) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This function is called by the :class:`grid2op.Chronics.FromHandlers` after the current handler has been created.
        Its goal is to initialize it with the relevant data from the environment.
        
        For example, if this handler represents "load_p" then `order_backend_arrays` will be the name of 
        each load in the environment and `names_chronics_to_backend` is a dictionnary mapping the name in the
        data to the names as read by the grid simulator / the backend.

        Parameters
        ----------
        order_backend_arrays : np.ndarray
            numpy array representing the name of the element in the grid
            
        names_chronics_to_backend : dict
            mapping between the names in `order_backend_arrays` and the names found in the data.
            
        """
        raise NotImplementedError()
    
    def done(self) -> bool:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        Whether or not this handler has done generating the data. It can be "done" in the case it 
        reads data from a csv and you are at the bottom line of the csv for example.

        Returns
        -------
        bool
            Whether it is "done" or not.
        """
        raise NotImplementedError()
    
    def load_next(self, dict_: dict) -> Optional[np.ndarray]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        
        This function is called by the :class:`grid2op.Chronics.FromHandlers`
        
        Used by the environment handlers ("load_p", "load_q", "prod_p" and "prod_v" only). When this function
        is called, it should return the next state of the type of data it is responsible for. If the previous
        state should not be modified, then this function can returns "None".
        
        This is called exactly once per step.

        Parameters
        ----------
        dict_ : dict
            A dictionnary representing the other "previous" data type. This function is always called in the same order:
            
            1) on "load_p"
            2) on "load_q"
            3) on "gen_p"
            4) on "gen_v"
            
            So if your handler is reponsible for "gen_p" then this dictionnary might contain 2 items:
            
            - key: "load_p", value: all the active loads for the environment at the same step (if the values are modified
              by the relevant handlers)
            - key: "load_q", value: all the reactive laods for the environment at the same step (if the values are modified
              by the relevant handlers)

        Returns
        -------
        Optional[np.ndarray]
            The new values (if any)
        """
        raise NotImplementedError()
    
    def check_validity(self, backend):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This function is called by the :class:`grid2op.Chronics.FromHandlers` after all the handlers have
        been initialized.
        
        Its role is to make sure that every handlers can "handle" the data of the environment smoothly. 
        
        It is called after each "env.reset()" call.

        Parameters
        ----------
        backend : :class:`grid2op.Backend.Backend`
            The backend used in the environment.
        """
        raise NotImplementedError()    
    
    def load_next_maintenance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This function is used only if the handler is reponsible for "maintenance". It is called
        exactly once per step.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            maintenance time: np.ndarray
                Time for next maintenance for each powerline
            
            maintenance duration: np.ndarray
                Duration of the next maintenance, for each powerline
                
        """
        raise NotImplementedError()
    
    def load_next_hazard(self) -> np.ndarray:
        # TODO
        raise NotImplementedError()
        
    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : "BaseHandler",  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple["BaseHandler", "BaseHandler", "BaseHandler", "BaseHandler"]
                 ) -> Optional[np.ndarray] :
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This function is called by the :class:`grid2op.Chronics.FromHandlers` only for the handlers responsible 
        for some "forecasts", which are "load_p_forecasted", "load_q_forecasted", "prod_p_forecasted", 
        "prod_v_forecasted".
        
        It is called exactly once per step and per horizon.

        It's similar to :func:`BaseHandler.load_next` with different inputs (because "forecast" are more complicated that 
        just real time data)
        
        .. seealso:: :func:`BaseHandler.load_next`
        
        Parameters
        ----------
        forecast_horizon_id : int
            The `id` of the horizon concerns. The horizon id is the index of the current horizon in the list :attr:`BaseHandler._h_forecast`
            
        inj_dict_env : dict
            The dictionnary containing the data of the environment (not the forecast) if data have been modified by the relevant handlers.
            
        inj_dict_previous_forecast : dict
            Similar to the `dict_` parameters of :func:`BaseHandler.load_next`
            
        env_handler : BaseHandler
            The handler of the same type as this one, but for the environment.
            
            For example, if this handler deals with "`load_q_forecasted`" then 
            `env_handler` will be the handler of `load_q`.
            
        env_handlers : Tuple[:class:`BaseHandler`, :class:`BaseHandler`, :class:`BaseHandler`, :class:`BaseHandler`]
            In these you have all the environment handlers in a tuple.
            
            The order is: "load_p", "load_q", "prod_p", "prod_v".

        Returns
        -------
        Optional[np.ndarray]
            The forecast (in the shape of numpy array) or None if nothing should be returned.
        """
        raise NotImplementedError()
    
    def get_future_data(self, horizon: int, quiet_warnings : bool=False) -> Optional[np.ndarray]:
        """        
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        
        This function is for example used in the 
        :class:`grid2op.Chronics.handlers.PerfectForecastHandler`: to generate a 
        "perfect forecast" this class will use this function to "have a look"
        into the future through this function.
        
        This function is for example implemented in 
        :class:`grid2op.Chronics.handlers.CSVHandler`

        Parameters
        ----------
        horizon : int
            The horizon (in minutes) to which we want the data.

        quiet_warnings: bool
            Whether to issue a warning (default, if quiet_warnings is False) or not
            
        Returns
        -------
        Optional[np.ndarray]
            The data that will be generated  in `horizon` minutes.
        """
        return None
    
    def next_chronics(self) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            
        This function is called by the :class:`grid2op.Chronics.FromHandlers` at the
        end of each episode when the next episode is loaded.
        """
        return None
