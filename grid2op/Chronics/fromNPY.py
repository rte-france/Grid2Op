# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional, Union, Dict, Literal, TYPE_CHECKING
import numpy as np
import hashlib
from datetime import datetime, timedelta

import grid2op
from grid2op.dtypes import dt_int
from grid2op.Chronics.gridValue import GridValue
from grid2op.Exceptions import ChronicsError

if TYPE_CHECKING:
    from grid2op.Backend import Backend   
    from grid2op.Action import PlayableAction 
    

class FromNPY(GridValue):
    """
    This class allows to generate some chronics compatible with grid2op if the data are provided in numpy format.

    It also enables the use of the starting the chronics at different time than the original time and to end it before the end
    of the chronics.

    It is then much more flexible in its usage than the defaults chronics. But it is also much more error prone. For example, it does not check
    the order of the loads / generators that you provide.

    .. warning::
        It assume the order of the elements are consistent with the powergrid backend ! It will not attempt to reorder the columns of the dataset

    .. note::

        The effect if "i_start" and "i_end" are persistant. If you set it once, it affects the object even after "env.reset()" is called.
        If you want to modify them, you need to use the :func:`FromNPY.chronics.change_i_start` and :func:`FromNPY.chronics.change_i_end` methods
        (and call `env.reset()`!)

    TODO implement methods to change the loads / production "based on sampling" (online sampling instead of only reading data)
    TODO implement the possibility to simulate maintenance / hazards "on the fly"
    TODO implement hazards !

    Examples
    --------

    Usage example, for what you don't really have to do:

    .. code-block:: python

        import grid2op
        from grid2op.Chronics import FromNPY

        # first retrieve the data that you want, the easiest wayt is to create an environment and read the data from it.
        env_name = "l2rpn_case14_sandbox"  # for example
        env_ref = grid2op.make(env_name)
        # retrieve the data
        load_p = 1.0 * env_ref.chronics_handler.real_data.data.load_p
        load_q = 1.0 * env_ref.chronics_handler.real_data.data.load_q
        prod_p = 1.0 * env_ref.chronics_handler.real_data.data.prod_p
        prod_v = 1.0 * env_ref.chronics_handler.real_data.data.prod_v

        # now create an environment with these chronics:
        env = grid2op.make(env_name,
                           chronics_class=FromNPY,
                           data_feeding_kwargs={"i_start": 5,  # start at the "step" 5 NB first step is first observation, available with `obs = env.reset()`
                                                "i_end": 18,  # end index: data after that will not be considered (excluded as per python convention)
                                                "load_p": load_p,
                                                "load_q": load_q,
                                                "prod_p": prod_p,
                                                "prod_v": prod_v
                                                ## other parameters includes
                                                # maintenance
                                                # load_p_forecast
                                                # load_q_forecast
                                                # prod_p_forecast
                                                # prod_v_forecast
                                                # init_state  # new in 1.10.2
                                                })

        # you can use env normally, including in runners
        obs = env.reset()
        # obs.load_p is load_p[5] (because you set "i_start" = 5, by default it's 0)

    You can, after creation, change the data with:

    .. code-block:: python

        # create env as above

        # retrieve some new values that you would like
        new_load_p = ...
        new_load_q = ...
        new_prod_p = ...
        new_prod_v = ...

        # change the values
        env.chronics_handler.real_data.change_chronics(new_load_p, new_load_q, new_prod_p, new_prod_v)
        obs = env.reset()  # mandatory if you want the change to be taken into account
        # obs.load_p is new_load_p[5]  (or rather load_p[env.chronics_handler.real_data._i_start])

    .. seealso::
        More usage examples in:

        - :func:`FromNPY.change_chronics`
        - :func:`FromNPY.change_forecasts`
        - :func:`FromNPY.change_i_start`
        - :func:`FromNPY.change_i_end`

    Attributes
    ----------
    TODO
    """
    MULTI_CHRONICS = False

    def __init__(
        self,
        load_p: np.ndarray,
        load_q: np.ndarray,
        prod_p: np.ndarray,
        prod_v: Optional[np.ndarray] = None,
        hazards: Optional[np.ndarray] = None,
        maintenance: Optional[np.ndarray] = None,
        load_p_forecast: Optional[np.ndarray] = None,  # TODO forecasts !!
        load_q_forecast: Optional[np.ndarray] = None,
        prod_p_forecast: Optional[np.ndarray] = None,
        prod_v_forecast: Optional[np.ndarray] = None,
        time_interval: timedelta = timedelta(minutes=5),
        max_iter: int = -1,
        start_datetime: datetime = datetime(year=2019, month=1, day=1),
        chunk_size: Optional[int] = None,
        i_start: Optional[int] = None,
        i_end: Optional[int] = None,  # excluded, as always in python
        init_state: Optional["grid2op.Action.BaseAction"] = None,
        **kwargs
    ):
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )
        self._i_start: int = i_start if i_start is not None else 0
        self.__new_istart: Optional[int] = i_start
        self.n_gen: int = prod_p.shape[1]
        self.n_load: int = load_p.shape[1]
        self.n_line: Union[int, None] = None

        self._load_p: np.ndarray = 1.0 * load_p
        self._load_q: np.ndarray = 1.0 * load_q
        self._prod_p: np.ndarray = 1.0 * prod_p

        self._prod_v = None
        if prod_v is not None:
            self._prod_v = 1.0 * prod_v

        self.__new_load_p: Optional[np.ndarray] = None
        self.__new_prod_p: Optional[np.ndarray] = None
        self.__new_prod_v: Optional[np.ndarray] = None
        self.__new_load_q: Optional[np.ndarray] = None

        self._i_end: int = i_end if i_end is not None else load_p.shape[0]
        self.__new_iend: Optional[int] = i_end

        self.has_maintenance = False
        self.maintenance = None
        self.maintenance_duration = None
        self.maintenance_time = None
        if maintenance is not None:
            self.has_maintenance = True
            self.n_line = maintenance.shape[1]
            if load_p.shape[0] != maintenance.shape[0]:
                raise ChronicsError("The maintenance file does not have the same number "
                                    "of time step at the load file")
            self.maintenance = maintenance  # TODO copy

            self.maintenance_time = (
                np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int)
                - 1
            )
            self.maintenance_duration = np.zeros(
                shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int
            )
            for line_id in range(self.n_line):
                self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(
                    self.maintenance[:, line_id]
                )
                self.maintenance_duration[
                    :, line_id
                ] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])

        self.has_hazards = False
        self.hazards = None
        self.hazard_duration = None
        if hazards is not None:
            raise ChronicsError(
                "This feature is not available at the moment. Fill a github issue at "
                "https://github.com/Grid2Op/grid2op/issues/new?assignees=&labels=enhancement&template=feature_request.md&title="
            )

        self._forecasts = None
        self._multi_forecasts = False
        self._load_p_forecast = None
        self._load_q_forecast = None
        self._prod_p_forecast = None
        self._prod_v_forecast = None
        self.n_forecast_horizons = 0

        self.__new_load_p_forecast = None
        self.__new_load_q_forecast = None
        self.__new_prod_p_forecast = None
        self.__new_prod_v_forecast = None
        self.__switch_to_multi_forecasts = False

        if load_p_forecast is not None:
            if load_q_forecast is None:
                raise ChronicsError("If you provide load_p_forecast you should provide load_q_forecast.")
            if prod_p_forecast is None:
                raise ChronicsError("If you provide load_p_forecast you should provide prod_p_forecast.")

            if load_p_forecast.ndim == 2:
                if load_q_forecast.ndim != 2 or prod_p_forecast.ndim != 2:
                    raise ChronicsError("All forecast arrays must use the same number of dimensions.")
                if prod_v_forecast is not None and prod_v_forecast.ndim != 2:
                    raise ChronicsError("All forecast arrays must use the same number of dimensions.")

                self._forecasts = FromNPY(
                    load_p=load_p_forecast,
                    load_q=load_q_forecast,
                    prod_p=prod_p_forecast,
                    prod_v=prod_v_forecast,
                    load_p_forecast=None,
                    load_q_forecast=None,
                    prod_p_forecast=None,
                    prod_v_forecast=None,
                    i_start=i_start,
                    i_end=i_end,
                )

            elif load_p_forecast.ndim == 3:
                self._multi_forecasts = True
                self._load_p_forecast = 1.0 * load_p_forecast
                self._load_q_forecast = 1.0 * load_q_forecast
                self._prod_p_forecast = 1.0 * prod_p_forecast

                if prod_v_forecast is not None:
                    self._prod_v_forecast = 1.0 * prod_v_forecast

                self._validate_multi_forecasts()
            else:
                raise ChronicsError(
                    "Forecast arrays must have shape (T, N) or (T, H, N)."
                )

        elif load_q_forecast is not None:
            raise ChronicsError(
                "if load_q_forecast is not None, then load_p_forecast should not be None"
            )
        elif prod_p_forecast is not None:
            raise ChronicsError(
                "if prod_p_forecast is not None, then load_p_forecast should not be None"
            )
        
        self._init_state = init_state
        self._max_iter = min(self._i_end - self._i_start, load_p.shape[0])

    @staticmethod
    def _format_multi_forecast_array(
        arr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if arr is None:
            return None

        if arr.ndim == 2:
            return 1.0 * arr[:, np.newaxis, :]

        if arr.ndim == 3:
            return 1.0 * arr

        raise ChronicsError(
            "Forecast arrays must have shape (T, N) or (T, H, N)."
        )

    def _validate_multi_forecasts(self) -> None:
        if self._load_p_forecast is None:
            raise ChronicsError(
                "load_p_forecast must be provided for multi-horizon forecasts."
            )

        if self._load_q_forecast is None:
            raise ChronicsError(
                "If you provide load_p_forecast you should provide load_q_forecast."
            )

        if self._prod_p_forecast is None:
            raise ChronicsError(
                "If you provide load_p_forecast you should provide prod_p_forecast."
            )

        expected_t = self._load_p.shape[0]
        expected_h = self._load_p_forecast.shape[1]

        arrays = (
            ("load_p_forecast", self._load_p_forecast, self.n_load),
            ("load_q_forecast", self._load_q_forecast, self.n_load),
            ("prod_p_forecast", self._prod_p_forecast, self.n_gen),
        )

        for name, arr, expected_n in arrays:
            if arr.ndim != 3:
                raise ChronicsError(
                    "All multi-horizon forecast arrays must have shape (T, H, N)."
                )

            if arr.shape[0] != expected_t:
                raise ChronicsError(
                    f"{name} must have the same number of timesteps as the chronics."
                )

            if arr.shape[1] != expected_h:
                raise ChronicsError(
                    "All forecast arrays must have the same number of horizons."
                )

            if arr.shape[2] != expected_n:
                raise ChronicsError(
                    f"{name} has an invalid number of columns for this environment."
                )

        if self._prod_v_forecast is not None:
            arr = self._prod_v_forecast

            if arr.ndim != 3:
                raise ChronicsError(
                    "All multi-horizon forecast arrays must have shape (T, H, N)."
                )

            if arr.shape[0] != expected_t:
                raise ChronicsError(
                    "prod_v_forecast must have the same number of timesteps as the chronics."
                )

            if arr.shape[1] != expected_h:
                raise ChronicsError(
                    "All forecast arrays must have the same number of horizons."
                )

            if arr.shape[2] != self.n_gen:
                raise ChronicsError(
                    "prod_v_forecast has an invalid number of columns for this environment."
                )

        self.n_forecast_horizons = expected_h

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        if len(order_backend_prods) != self.n_gen:
            raise ChronicsError(f"len(order_backend_prods)={len(order_backend_prods)} vs self.n_gen={self.n_gen}")
        
        if len(order_backend_loads) != self.n_load:
            raise ChronicsError(f"len(order_backend_loads)={len(order_backend_loads)} vs self.n_load={self.n_load}")
        
        if self.n_line is None:
            self.n_line = len(order_backend_lines)
        else:
            if len(order_backend_lines) != self.n_line:
                raise ChronicsError(f"len(order_backend_lines)={len(order_backend_lines)} vs self.n_line={self.n_line}")

        if self._forecasts is not None:
            self._forecasts.initialize(
                order_backend_loads,
                order_backend_prods,
                order_backend_lines,
                order_backend_subs,
                names_chronics_to_backend,
            )
        self.maintenance_time_nomaint = np.zeros(shape=(self.n_line,), dtype=dt_int) - 1
        self.maintenance_duration_nomaint = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.hazard_duration_nohaz = np.zeros(shape=(self.n_line,), dtype=dt_int)

        self.curr_iter = 0
        self.current_index = self._i_start - 1
        self._max_iter = self._i_end - self._i_start

    def _get_long_hash(self, hash_: Optional[hashlib.blake2b] = None):
        # get the "long hash" from blake2b
        if hash_ is None:
            hash_ = (
                hashlib.blake2b()
            )  # should be faster than md5 ! (and safer, but we only care about speed here)
        hash_.update(self._load_p.tobytes())
        hash_.update(self._load_q.tobytes())
        hash_.update(self._prod_p.tobytes())
        if self._prod_v is not None:
            hash_.update(self._prod_v.tobytes())
        if self.maintenance is not None:
            hash_.update(self.maintenance.tobytes())
        if self.hazards is not None:
            hash_.update(self.hazards.tobytes())

        if self._forecasts:
            self._forecasts._get_long_hash(hash_)

        if self._multi_forecasts:
            for arr in (
                self._load_p_forecast,
                self._load_q_forecast,
                self._prod_p_forecast,
                self._prod_v_forecast,
            ):
                if arr is not None:
                    hash_.update(arr.tobytes())

        return hash_.digest()

    def get_id(self) -> str:
        """
        To return a unique ID of the chronics, we use a hash function (black2b), but it outputs a name too big (64 characters or so).
        So we hash it again with md5 to get a reasonable length id (32 characters)

        Returns:
            str:  the hash of the arrays (load_p, load_q, etc.) in the chronics
        """
        long_hash_byte = self._get_long_hash()
        # now shorten it with md5
        short_hash = hashlib.md5(long_hash_byte)
        return short_hash.hexdigest()

    @staticmethod
    def _create_dict_inj(res, obj_with_inj_data):
        dict_ = {}
        prod_v = None
        if obj_with_inj_data._load_p is not None:
            dict_["load_p"] = 1.0 * obj_with_inj_data._load_p[obj_with_inj_data.current_index, :]
        if obj_with_inj_data._load_q is not None:
            dict_["load_q"] = 1.0 * obj_with_inj_data._load_q[obj_with_inj_data.current_index, :]
            
        array_gen_p = obj_with_inj_data._gen_p if hasattr(obj_with_inj_data, "_gen_p") else obj_with_inj_data._prod_p
        if array_gen_p is not None:
            dict_["prod_p"] = 1.0 * array_gen_p[obj_with_inj_data.current_index, :]
            
        array_gen_v = obj_with_inj_data._gen_v if hasattr(obj_with_inj_data, "_gen_v") else obj_with_inj_data._prod_v
        if array_gen_v is not None:
            prod_v = 1.0 * array_gen_v[obj_with_inj_data.current_index, :]
            
        if dict_:
            res["injection"] = dict_
        return prod_v
    
    @staticmethod
    def _create_dict_maintenance_hazards(res, obj_with_inj_data):
        if obj_with_inj_data.maintenance is not None and obj_with_inj_data.has_maintenance:
            res["maintenance"] = obj_with_inj_data.maintenance[obj_with_inj_data.current_index, :]
        if obj_with_inj_data.hazards is not None and obj_with_inj_data.has_hazards:
            res["hazards"] = obj_with_inj_data.hazards[obj_with_inj_data.current_index, :]
            
        if (
            obj_with_inj_data.maintenance_time is not None
            and obj_with_inj_data.maintenance_duration is not None
            and obj_with_inj_data.has_maintenance
        ):
            maintenance_time = dt_int(1 * obj_with_inj_data.maintenance_time[obj_with_inj_data.current_index, :])
            maintenance_duration = dt_int(
                1 * obj_with_inj_data.maintenance_duration[obj_with_inj_data.current_index, :]
            )
        else:
            maintenance_time = obj_with_inj_data.maintenance_time_nomaint
            maintenance_duration = obj_with_inj_data.maintenance_duration_nomaint

        if obj_with_inj_data.hazard_duration is not None and obj_with_inj_data.has_hazards:
            hazard_duration = 1 * obj_with_inj_data.hazard_duration[obj_with_inj_data.current_index, :]
        else:
            hazard_duration = obj_with_inj_data.hazard_duration_nohaz
        return maintenance_time, maintenance_duration, hazard_duration
    
    def load_next(self):
        self.current_index += 1

        if (
            self.current_index > self._i_end
            or self.current_index >= self._load_p.shape[0]
        ):
            raise StopIteration

        res = {}
        prod_v = FromNPY._create_dict_inj(res, self)
        maintenance_time, maintenance_duration, hazard_duration = FromNPY._create_dict_maintenance_hazards(res, self)

        self.current_datetime += self.time_interval
        self.curr_iter += 1

        return (
            self.current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            hazard_duration,
            prod_v,
        )

    def check_validity(
        self, backend: Optional["Backend"]
    ) -> None:
        if self._load_p.shape[0] != self._load_q.shape[0]:
            raise ChronicsError("Loads P and Q must have the same number of timesteps / rows")
        if self._load_p.shape[0] != self._prod_p.shape[0]:
            raise ChronicsError("Loads P and gen p must have the same number of timesteps / rows")
        if self._prod_v is not None:
            if self._load_p.shape[0] != self._prod_v.shape[0]:
                raise ChronicsError("Loads P and gen v (as it is provided) must have the same number of timesteps / rows")

        if self.hazards is not None:
            if self.hazards.shape[1] != self.n_line:
                raise ChronicsError("Loads P and hazards (as it is provided) must have the same number of timesteps / rows")
                
        if self.maintenance is not None:
            if self.maintenance.shape[1] != self.n_line:
                raise ChronicsError("Maintenance file (as it is provided) should have the "
                                    "same number of columns as the number of powerlines on the grid")
        if self.maintenance_duration is not None:
            if self.n_line != self.maintenance_duration.shape[1]:
                raise ChronicsError("maintenance_duration file (as it is provided) should have the same "
                                    "number of columns as the number of powerlines on the grid")
        if self.maintenance_time is not None:
            if self.n_line != self.maintenance_time.shape[1]:
                raise ChronicsError("maintenance_time file (as it is provided) should have the same "
                                    "number of columns as the number of powerlines on the grid")

        # TODO forecast
        if self._forecasts is not None:
            if self._forecasts.n_line != self.n_line:
                raise ChronicsError("self._forecasts.n_line != self.n_line")
            if self._forecasts.n_gen != self.n_gen:
                raise ChronicsError("self._forecasts.n_gen != self.n_gen")
            if self._forecasts.n_load != self.n_load:
                raise ChronicsError("self._forecasts.n_load != self.n_load")
            if self._load_p.shape[0] != self._forecasts._load_p.shape[0]:
                raise ChronicsError("self._load_p.shape[0] != self._forecasts._load_p.shape[0]")
            if self._load_q.shape[0] != self._forecasts._load_q.shape[0]:
                raise ChronicsError("self._load_q.shape[0] != self._forecasts._load_q.shape[0]")
            if self._prod_p.shape[0] != self._forecasts._prod_p.shape[0]:
                raise ChronicsError("self._prod_p.shape[0] != self._forecasts._prod_p.shape[0]")
            if self._prod_v is not None and self._forecasts._prod_v is not None:
                if self._prod_v.shape[0] != self._forecasts._prod_v.shape[0]:
                    raise ChronicsError("self._prod_v.shape[0] != self._forecasts._prod_v.shape[0]")
            self._forecasts.check_validity(backend=backend)

        if self._multi_forecasts:
            self._validate_multi_forecasts()

    def next_chronics(self):
        # restart the chronics: read it again !
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
        if self.__new_istart is not None:
            self._i_start = self.__new_istart
        else:
            self._i_start = 0
        self.current_index = self._i_start

        if self.__new_load_p is not None:
            self._load_p = self.__new_load_p
            self.__new_load_p = None
        if self.__new_load_q is not None:
            self._load_q = self.__new_load_q
            self.__new_load_q = None
        if self.__new_prod_p is not None:
            self._prod_p = self.__new_prod_p
            self.__new_prod_p = None
        if self.__new_prod_v is not None:
            self._prod_v = self.__new_prod_v
            self.__new_prod_v = None

        if self.__new_iend is None:
            self._i_end = self._load_p.shape[0]
        else:
            self._i_end = self.__new_iend

        if self.__switch_to_multi_forecasts:
            self._forecasts = None
            self._multi_forecasts = True

            self._load_p_forecast = self.__new_load_p_forecast
            self._load_q_forecast = self.__new_load_q_forecast
            self._prod_p_forecast = self.__new_prod_p_forecast
            self._prod_v_forecast = self.__new_prod_v_forecast

            self.__new_load_p_forecast = None
            self.__new_load_q_forecast = None
            self.__new_prod_p_forecast = None
            self.__new_prod_v_forecast = None
            self.__switch_to_multi_forecasts = False

            self._validate_multi_forecasts()

        elif self._multi_forecasts:
            if self.__new_load_p_forecast is not None:
                self._load_p_forecast = self.__new_load_p_forecast
                self.__new_load_p_forecast = None

            if self.__new_load_q_forecast is not None:
                self._load_q_forecast = self.__new_load_q_forecast
                self.__new_load_q_forecast = None

            if self.__new_prod_p_forecast is not None:
                self._prod_p_forecast = self.__new_prod_p_forecast
                self.__new_prod_p_forecast = None

            if self.__new_prod_v_forecast is not None:
                self._prod_v_forecast = self.__new_prod_v_forecast
                self.__new_prod_v_forecast = None

            self._validate_multi_forecasts()

        elif self._forecasts is not None:
            self._forecasts.next_chronics()

        self.check_validity(backend=None)
        self._max_iter = self._i_end - self._i_start

    def done(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compare to :func:`GridValue.done` an episode can be over for 2 main reasons:

          - :attr:`GridValue.max_iter` has been reached
          - There are no data in the numpy array.
          - i_end has been reached

        The episode is done if one of the above condition is met.

        Returns
        -------
        res: ``bool``
            Whether the episode has reached its end or not.

        """
        res = False
        if (
            self.current_index >= self._i_end
            or self.current_index >= self._load_p.shape[0]
        ):
            res = True
        elif self._max_iter > 0:
            if self.curr_iter > self._max_iter:
                res = True
        return res

    def forecasts(self):
        """
        Return forecasts available from the current timestep.

        Existing two-dimensional forecast arrays keep the historical
        one-step-ahead behaviour. Three-dimensional arrays use the shape
        ``(T, H, N)`` and return one entry per horizon.
        """
        if self._multi_forecasts:
            if (
                self.current_index < 0
                or self.current_index >= self._load_p_forecast.shape[0]
            ):
                return []

            results = []

            for h in range(self.n_forecast_horizons):
                dict_ = {
                    "load_p": 1.0
                    * self._load_p_forecast[self.current_index, h, :],
                    "load_q": 1.0
                    * self._load_q_forecast[self.current_index, h, :],
                    "prod_p": 1.0
                    * self._prod_p_forecast[self.current_index, h, :],
                }

                if self._prod_v_forecast is not None:
                    dict_["prod_v"] = (
                        1.0
                        * self._prod_v_forecast[self.current_index, h, :]
                    )

                results.append(
                    (
                        self.current_datetime
                        + (h + 1) * self.time_interval,
                        {"injection": dict_},
                    )
                )

            return results

        if self._forecasts is None:
            return []

        self._forecasts.current_index = self.current_index - 1
        dt, dict_, *rest = self._forecasts.load_next()

        return [
            (
                self.current_datetime + self.time_interval,
                dict_,
            )
        ]

    def change_chronics(
        self,
        new_load_p: np.ndarray = None,
        new_load_q: np.ndarray = None,
        new_prod_p: np.ndarray = None,
        new_prod_v: np.ndarray = None,
    ):
        """
        Allows to change the data used by this class.

        .. warning::
            This has an effect only after "env.reset" has been called !


        Args:
            new_load_p (np.ndarray, optional): change the load_p. Defaults to None (= do not change).
            new_load_q (np.ndarray, optional): change the load_q. Defaults to None (= do not change).
            new_prod_p (np.ndarray, optional): change the prod_p. Defaults to None (= do not change).
            new_prod_v (np.ndarray, optional): change the prod_v. Defaults to None (= do not change).

        Examples
        ---------

        .. code-block:: python

            import grid2op
            from grid2op.Chronics import FromNPY
            # create an environment as in this class description (in short: )

            load_p = ...  # find somehow a suitable "load_p" array: rows represent time, columns the individual load
            load_q = ...
            prod_p = ...
            prod_v = ...

            # now create an environment with these chronics:
            env = grid2op.make(env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"load_p": load_p,
                                                    "load_q": load_q,
                                                    "prod_p": prod_p,
                                                    "prod_v": prod_v}
                               )
            obs = env.reset()  # obs.load_p is load_p[0] (or rather load_p[env.chronics_handler.real_data._i_start])

            new_load_p = ...  # find somehow a new suitable "load_p"
            new_load_q = ...
            new_prod_p = ...
            new_prod_v = ...

            env.chronics_handler.real_data.change_chronics(new_load_p, new_load_q, new_prod_p, new_prod_v)
            # has no effect at this stage

            obs = env.reset()  # now has some effect !
            # obs.load_p is new_load_p[0]  (or rather load_p[env.chronics_handler.real_data._i_start])
        """
        if new_load_p is not None:
            self.__new_load_p = 1.0 * new_load_p
        if new_load_q is not None:
            self.__new_load_q = 1.0 * new_load_q
        if new_prod_p is not None:
            self.__new_prod_p = 1.0 * new_prod_p
        if new_prod_v is not None:
            self.__new_prod_v = 1.0 * new_prod_v

    def change_forecasts(
        self,
        new_load_p: np.ndarray = None,
        new_load_q: np.ndarray = None,
        new_prod_p: np.ndarray = None,
        new_prod_v: np.ndarray = None,
    ):
        """
        Allows to change the data used by this class in the "obs.simulate" function.

        .. warning::
            This has an effect only after "env.reset" has been called !

        Args:
            new_load_p (np.ndarray, optional): change the load_p_forecast. Defaults to None (= do not change).
            new_load_q (np.ndarray, optional): change the load_q_forecast. Defaults to None (= do not change).
            new_prod_p (np.ndarray, optional): change the prod_p_forecast. Defaults to None (= do not change).
            new_prod_v (np.ndarray, optional): change the prod_v_forecast. Defaults to None (= do not change).

        Examples
        ---------

        .. code-block:: python

            import grid2op
            from grid2op.Chronics import FromNPY
            # create an environment as in this class description (in short: )

            load_p = ...  # find somehow a suitable "load_p" array: rows represent time, columns the individual load
            load_q = ...
            prod_p = ...
            prod_v = ...
            load_p_forecast = ...
            load_q_forecast = ...
            prod_p_forecast = ...
            prod_v_forecast = ...

            env = grid2op.make(env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"load_p": load_p,
                                                    "load_q": load_q,
                                                    "prod_p": prod_p,
                                                    "prod_v": prod_v,
                                                    "load_p_forecast": load_p_forecast
                                                    "load_q_forecast": load_q_forecast
                                                    "prod_p_forecast": prod_p_forecast
                                                    "prod_v_forecast": prod_v_forecast
                                                    })

            new_load_p_forecast = ...  # find somehow a new suitable "load_p"
            new_load_q_forecast = ...
            new_prod_p_forecast = ...
            new_prod_v_forecast = ...

            env.chronics_handler.real_data.change_forecasts(new_load_p_forecast, new_load_q_forecast, new_prod_p_forecast, new_prod_v_forecast)
            # has no effect at this stage

            obs = env.reset()  # now has some effect !
            sim_o, *_ = obs.simulate()  # sim_o.load_p has the values of new_load_p_forecast[0]
        """
        requested_multi = (
            (new_load_p is not None and new_load_p.ndim == 3)
            or (new_load_q is not None and new_load_q.ndim == 3)
            or (new_prod_p is not None and new_prod_p.ndim == 3)
            or (new_prod_v is not None and new_prod_v.ndim == 3)
        )

        if self._multi_forecasts or requested_multi:
            if (
                new_load_p is None
                or new_load_q is None
                or new_prod_p is None
            ):
                raise ChronicsError(
                    "load_p, load_q and prod_p forecasts must all be "
                    "provided when switching to multi-horizon forecasts."
                )

            self.__new_load_p_forecast = (
                self._format_multi_forecast_array(new_load_p)
            )
            self.__new_load_q_forecast = (
                self._format_multi_forecast_array(new_load_q)
            )
            self.__new_prod_p_forecast = (
                self._format_multi_forecast_array(new_prod_p)
            )
            self.__new_prod_v_forecast = (
                self._format_multi_forecast_array(new_prod_v)
            )

            if requested_multi and not self._multi_forecasts:
                self.__switch_to_multi_forecasts = True

            return

        if self._forecasts is None:
            raise ChronicsError(
                "You cannot change the forecast for this chronics are there are no forecasts enabled"
            )

        self._forecasts.change_chronics(
            new_load_p=new_load_p,
            new_load_q=new_load_q,
            new_prod_p=new_prod_p,
            new_prod_v=new_prod_v,
        )

    def max_timestep(self):
        if self._max_iter >= 0:
            return min(self._max_iter, self._load_p.shape[0], self._i_end)
        return min(self._load_p.shape[0], self._i_end)

    def change_i_start(self, new_i_start: Union[int, None]):
        """
        Allows to change the "i_start".

        .. warning::

            It has only an affect after "env.reset()" is called.

        Examples
        --------

        .. code-block:: python

            import grid2op
            from grid2op.Chronics import FromNPY
            # create an environment as in this class description (in short: )

            load_p = ...  # find somehow a suitable "load_p" array: rows represent time, columns the individual load
            load_q = ...
            prod_p = ...
            prod_v = ...

            # now create an environment with these chronics:
            env = grid2op.make(env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"load_p": load_p,
                                                    "load_q": load_q,
                                                    "prod_p": prod_p,
                                                    "prod_v": prod_v}
                               )
            obs = env.reset()  # obs.load_p is load_p[0] (or rather load_p[env.chronics_handler.real_data._i_start])

            env.chronics_handler.real_data.change_i_start(10)
            obs = env.reset()  # obs.load_p is load_p[10]
            # indeed `env.chronics_handler.real_data._i_start` has been changed to 10.

            # to undo all changes (and use the defaults) you can:
            # env.chronics_handler.real_data.change_i_start(None)
        """
        if new_i_start is not None:
            self.__new_istart = int(new_i_start)
        else:
            self.__new_istart = None
            

    def change_i_end(self, new_i_end: Union[int, None]):
        """
        Allows to change the "i_end".

        .. warning::

            It has only an affect after "env.reset()" is called.

        Examples
        --------

        .. code-block:: python

            import grid2op
            from grid2op.Chronics import FromNPY
            # create an environment as in this class description (in short: )

            load_p = ...  # find somehow a suitable "load_p" array: rows represent time, columns the individual load
            load_q = ...
            prod_p = ...
            prod_v = ...

            # now create an environment with these chronics:
            env = grid2op.make(env_name,
                               chronics_class=FromNPY,
                               data_feeding_kwargs={"load_p": load_p,
                                                    "load_q": load_q,
                                                    "prod_p": prod_p,
                                                    "prod_v": prod_v}
                               )
            obs = env.reset()

            env.chronics_handler.real_data.change_i_end(150)
            obs = env.reset()
            # indeed `env.chronics_handler.real_data._i_end` has been changed to 10.
            # scenario lenght will be at best 150 !

            # to undo all changes (and use the defaults) you can:
            # env.chronics_handler.real_data.change_i_end(None)
        """
        if new_i_end is not None:
            self.__new_iend = int(new_i_end)
        else:
            self.__new_iend = None

    def get_init_action(self,
                        names_chronics_to_backend: Optional[Dict[Literal["loads", "prods", "lines"],
                                                                 Dict[str, str]]]=None) -> Union["PlayableAction", None]:
        # names_chronics_to_backend is ignored, names should be consistent between the environment 
        # and the initial state
        
        return self._init_state
