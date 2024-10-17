# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This class abstracts the main components of BaseAction, BaseObservation, ActionSpace, and ObservationSpace.

It represents a powergrid (the object in it) in a format completely agnostic to the solver used to compute
the power flows (:class:`grid2op.Backend.Backend`).

See :class:`grid2op.Converter` for a different type of Action / Observation. These can be used to transform
complex :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observaion` into more convient structures
to manipulate.

"""
import warnings
import copy
import os
import numpy as np
import sys
from packaging import version
from typing import Dict, Union, Literal, Any, List, Optional, ClassVar, Tuple
    
import grid2op
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.typing_variables import CLS_AS_DICT_TYPING, N_BUSBAR_PER_SUB_TYPING
from grid2op.Exceptions import *
from grid2op.Space.space_utils import extract_from_dict, save_to_dict

# TODO tests of these methods and this class in general
DEFAULT_N_BUSBAR_PER_SUB = 2


class GridObjects:
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        Almost every class inherit from this class, so they have its methods and attributes.
        Do not attempt to use it outside of grid2op environment.

    This class stores in a "Backend agnostic" way some information about the powergrid. All these attributes
    are constant throughout an episode and are defined when the backend is loaded by the environment.

    It stores information about numbers of objects, and which objects are where, their names, etc.

    The classes :class:`grid2op.Action.BaseAction`, :class:`grid2op.Action.ActionSpace`,
    :class:`grid2op.Observation.BaseObservation`, :class:`grid2op.Observation.ObservationSpace` and
    :class:`grid2op.Backend.Backend` all inherit from this class. This means that each of the above has its own
    representation of the powergrid.

    Before diving into the technical details on the implementation, you might want to have a look at this
    page of the documentation :ref:`graph-encoding-gridgraph` that details why this representation is suitable.

    The modeling adopted for describing a powergrid is the following:

    - only the main objects of a powergrid are represented. An "object" is either a load (consumption) a generator
      (production), an end of a powerline (each powerline have exactly two extremities: "origin" (or)
      and "extremity" (ext)).
    - every "object" (see above) is connected to a unique substation. Each substation then counts a given (fixed)
      number of objects connected to it. [in this platform we don't consider the possibility to build new "objects" as
      of today]

    For each object, the bus to which it is connected is given in the `*_to_subid` (for
    example :attr:`GridObjects.load_to_subid` gives, for each load, the id of the substation to which it is
    connected)

    We suppose that, at every substation, each object (if connected) can be connected to either "busbar" 1 or
    "busbar" 2. This means that, at maximum, there are 2 independent buses for each substation.

    With this hypothesis, we can represent (thought experiment) each substation by a vector. This vector has as
    many components than the number of objects in the substation (following the previous example, the vector
    representing the first substation would have 5 components). And each component of this vector would represent
    a fixed element in it. For example, if say, the load with id 1 is connected to the first element, there would be
    a unique component saying if the load with id 1 is connected to busbar 1 or busbar 2. For the generators, this
    id in this (fictive) vector is indicated in the :attr:`GridObjects.gen_to_sub_pos` vector. For example the first
    position of :attr:`GridObjects.gen_to_sub_pos` indicates on which component of the (fictive) vector representing
    the
    substation 1 to look to know on which bus the first generator is connected.

    We define the "topology" as the busbar to which each object is connected: each object being connected to either
    busbar 1 or busbar 2, this topology can be represented by a vector of fixed size (and it actually is in
    :attr:`grid2op.Observation.BaseObservation.topo_vect` or in :func:`grid2op.Backend.Backend.get_topo_vect`).
    There are
    multiple ways to make such a vector. We decided to concatenate all the (fictive) vectors described above. This
    concatenation represents the actual topology of this powergrid at a given timestep. This class doesn't store this
    information (see :class:`grid2op.Observation.BaseObservation` for such purpose).
    This entails that:

    - the bus to which each object on a substation will be stored in consecutive components of such a vector. For
      example, if the first substation of the grid has 5 elements connected to it, then the first 5 elements of
      :attr:`grid2op.Observation.BaseObservation.topo_vect` will represent these 5 elements. The number of elements
      in each substation is given in :attr:`grid2op.Space.GridObjects.sub_info`.
    - the substation are stored in "order": objects of the first substations are represented, then this is the objects
      of the second substation etc. So in the example above, the 6th element of
      :attr:`grid2op.Observation.BaseObservation.topo_vect` is an object connected to the second substation.
    - to know on which position of this "topology vector" we can find the information relative a specific element
      it is possible to:

        - method 1 (not recommended):

          i) retrieve the substation to which this object is connected (for example looking at
             :attr:`GridObjects.line_or_to_subid` [l_id] to know on which substation is connected the origin of
             powerline with id $l_id$.)
          ii) once this substation id is known, compute which are the components of the topological vector that encodes
              information about this substation. For example, if the substation id `sub_id` is 4, we a) count the number
              of elements in substations with id 0, 1, 2 and 3 (say it's 42) we know, by definition that the substation
              4 is encoded in ,:attr:`grid2op.Observation.BaseObservation.topo_vect` starting at component 42 and b)
              this
              substations has :attr:`GridObjects.sub_info` [sub_id] elements (for the sake of the example say it's 5)
              then the end of the vector for substation 4 will be 42+5 = 47. Finally, we got the representation of the
              "local topology" of the substation 4 by looking at
              :attr:`grid2op.Observation.BaseObservation.topo_vect` [42:47].
          iii) retrieve which component of this vector of dimension 5 (remember we assumed substation 4 had 5 elements)
               encodes information about the origin side of the line with id `l_id`. This information is given in
               :attr:`GridObjects.line_or_to_sub_pos` [l_id]. This is a number between 0 and 4, say it's 3. 3 being
               the index of the object in the substation)

        - method 2 (not recommended): all of the above is stored (for the same powerline) in the
          :attr:`GridObjects.line_or_pos_topo_vect` [l_id]. In the example above, we will have:
          :attr:`GridObjects.line_or_pos_topo_vect` [l_id] = 45 (=42+3):
          42 being the index on which the substation started and 3 being the index of the object in the substation)
        - method 3 (recommended): use any of the function that computes it for you:
          :func:`grid2op.Observation.BaseObservation.state_of` is such an interesting method. The two previous methods
          "method 1" and "method 2" were presented as a way to give detailed and "concrete" example on how the
          modeling of the powergrid work.
        - method 4 (recommended): use the :func:`GridObjects.topo_vect_element`

    For a given powergrid, this object should be initialized once in the :class:`grid2op.Backend.Backend` when
    the first call to :func:`grid2op.Backend.Backend.load_grid` is performed. In particular the following attributes
    must necessarily be defined (see above for a detailed description of some of the attributes):

    - :attr:`GridObjects.name_load`
    - :attr:`GridObjects.name_gen`
    - :attr:`GridObjects.name_line`
    - :attr:`GridObjects.name_sub`
    - :attr:`GridObjects.name_storage`
    - :attr:`GridObjects.n_line`
    - :attr:`GridObjects.n_gen`
    - :attr:`GridObjects.n_load`
    - :attr:`GridObjects.n_sub`
    - :attr:`GridObjects.n_storage`
    - :attr:`GridObjects.sub_info`
    - :attr:`GridObjects.dim_topo`
    - :attr:`GridObjects.load_to_subid`
    - :attr:`GridObjects.gen_to_subid`
    - :attr:`GridObjects.line_or_to_subid`
    - :attr:`GridObjects.line_ex_to_subid`
    - :attr:`GridObjects.storage_to_subid`

    Optionally, to have more control on the internal grid2op representation, you can also set:

    - :attr:`GridObjects.load_to_sub_pos`
    - :attr:`GridObjects.gen_to_sub_pos`
    - :attr:`GridObjects.line_or_to_sub_pos`
    - :attr:`GridObjects.line_ex_to_sub_pos`
    - :attr:`GridObjects.storage_to_sub_pos`

    A call to the function :func:`GridObjects._compute_pos_big_topo_cls` allow to compute the \\*_pos_topo_vect attributes
    (for example :attr:`GridObjects.line_ex_pos_topo_vect`) can be computed from the above data:

    - :attr:`GridObjects.load_pos_topo_vect`
    - :attr:`GridObjects.gen_pos_topo_vect`
    - :attr:`GridObjects.line_or_pos_topo_vect`
    - :attr:`GridObjects.line_ex_pos_topo_vect`
    - :attr:`GridObjects.storage_pos_topo_vect`


    Note that if you want to model an environment with unit commitment or redispatching capabilities, you also need
    to provide the following attributes:

    - :attr:`GridObjects.gen_type`
    - :attr:`GridObjects.gen_pmin`
    - :attr:`GridObjects.gen_pmax`
    - :attr:`GridObjects.gen_redispatchable`
    - :attr:`GridObjects.gen_max_ramp_up`
    - :attr:`GridObjects.gen_max_ramp_down`
    - :attr:`GridObjects.gen_min_uptime`
    - :attr:`GridObjects.gen_min_downtime`
    - :attr:`GridObjects.gen_cost_per_MW`
    - :attr:`GridObjects.gen_startup_cost`
    - :attr:`GridObjects.gen_shutdown_cost`
    - :attr:`GridObjects.gen_renewable`

    These information are loaded using the :func:`grid2op.Backend.Backend.load_redispatching_data` method.

    Note that if you want to model an environment with flexibility capabilities, you also need
    to provide the following attributes:

    - :attr:`GridObjects.load_size`
    - :attr:`GridObjects.load_redispatchable`
    - :attr:`GridObjects.load_max_ramp_up`
    - :attr:`GridObjects.load_max_ramp_down`
    - :attr:`GridObjects.load_min_uptime`
    - :attr:`GridObjects.load_min_downtime`
    - :attr:`GridObjects.load_cost_per_MW`

    These information are loaded using the :func:`grid2op.Backend.Backend.load_flexibility_data` method.

    **NB** it does not store any information about the current state of the powergrid. It stores information that
    cannot be modified by the BaseAgent, the Environment or any other entity.

    Attributes
    ----------

    n_busbar_per_sub: :class:`int`
        number of independant busbars for all substations [*class attribute*]. It's 2 by default
        or if the implementation of the backend does not support this feature.
        
        .. versionadded:: 1.10.0

    n_line: :class:`int`
        number of powerlines in the powergrid [*class attribute*]

    n_gen: :class:`int`
        number of generators in the powergrid [*class attribute*]

    n_load: :class:`int`
        number of loads in the powergrid. [*class attribute*]

    n_sub: :class:`int`
        number of substations in the powergrid. [*class attribute*]

    n_storage: :class:`int`
        number of storage units in the powergrid. [*class attribute*]

    dim_topo: :class:`int`
        The total number of objects in the powergrid.
        This is also the dimension of the "topology vector" defined above. [*class attribute*]

    sub_info: :class:`numpy.ndarray`, dtype:int
        for each substation, gives the number of elements connected to it [*class attribute*]

    load_to_subid: :class:`numpy.ndarray`, dtype:int
        for each load, gives the id the substation to which it is connected. For example,
        :attr:`GridObjects.load_to_subid` [load_id] gives the id of the substation to which the load of id
        `load_id` is connected. [*class attribute*]

    gen_to_subid: :class:`numpy.ndarray`, dtype:int
        for each generator, gives the id the substation to which it is connected [*class attribute*]

    line_or_to_subid: :class:`numpy.ndarray`, dtype:int
        for each line, gives the id the substation to which its "origin" end is connected [*class attribute*]

    line_ex_to_subid: :class:`numpy.ndarray`, dtype:int
        for each line, gives the id the substation to which its "extremity" end is connected [*class attribute*]

    storage_to_subid: :class:`numpy.ndarray`, dtype:int
        for each storage unit, gives the id the substation to which it is connected [*class attribute*]

    load_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        Suppose you represent the topoology of the substation *s* with a vector (each component of this vector will
        represent an object connected to this substation). This vector has, by definition the size
        :attr:`GridObject.sub_info` [s]. `load_to_sub_pos` tells which component of this vector encodes the
        current load. Suppose that load of id `l` is connected to the substation of id `s` (this information is
        stored in :attr:`GridObjects.load_to_subid` [l]), then if you represent the topology of the substation
        `s` with a vector `sub_topo_vect`, then "`sub_topo_vect` [ :attr:`GridObjects.load_to_subid` [l] ]" will encode
        on which bus the load of id `l` is stored. [*class attribute*]

    gen_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for generators. [*class attribute*]

    line_or_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos`  but for "origin" end of powerlines. [*class attribute*]

    line_ex_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for "extremity" end of powerlines. [*class attribute*]

    storage_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for storage units. [*class attribute*]

    load_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        The topology if the entire grid is given by a vector, say *topo_vect* of size
        :attr:`GridObjects.dim_topo`. For a given load of id *l*,
        :attr:`GridObjects.load_to_sub_pos` [l] is the index
        of the load *l* in the vector :attr:`grid2op.BaseObservation.BaseObservation.topo_vect` .
        This means that, if
        "`topo_vect` [ :attr:`GridObjects.load_pos_topo_vect` \\[l\\] ]=2"
        then load of id *l* is connected to the second bus of the substation. [*class attribute*]

    gen_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for generators. [*class attribute*]

    line_or_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for "origin" end of powerlines. [*class attribute*]

    line_ex_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for "extremity" end of powerlines. [*class attribute*]

    storage_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for storage units. [*class attribute*]

    name_load: :class:`numpy.ndarray`, dtype:str
        ordered names of the loads in the grid. [*class attribute*]

    name_gen: :class:`numpy.ndarray`, dtype:str
        ordered names of the productions in the grid. [*class attribute*]

    name_line: :class:`numpy.ndarray`, dtype:str
        ordered names of the powerline in the grid. [*class attribute*]

    name_sub: :class:`numpy.ndarray`, dtype:str
        ordered names of the substation in the grid [*class attribute*]

    name_storage: :class:`numpy.ndarray`, dtype:str
        ordered names of the storage units in the grid [*class attribute*]

    attr_list_vect: ``list``, static
        List of string. It represents the attributes that will be stored to/from vector when the BaseObservation is
        converted
        to/from it. This parameter is also used to compute automatically :func:`GridObjects.dtype` and
        :func:`GridObjects.shape` as well as :func:`GridObjects.size`. If this class is derived, then it's really
        important that this vector is properly set. All the attributes with the name on this vector should have
        consistently the same size and shape, otherwise, some methods will not behave as expected. [*class attribute*]

    _vectorized: :class:`numpy.ndarray`, dtype:float
        The representation of the GridObject as a vector. See the help of :func:`GridObjects.to_vect` and
        :func:`GridObjects.from_vect` for more information. **NB** for performance reason, the conversion of the
        internal
        representation to a vector is not performed at any time. It is only performed when :func:`GridObjects.to_vect`
        is
        called the first time. Otherwise, this attribute is set to ``None``. [*class attribute*]

    gen_type: :class:`numpy.ndarray`, dtype:str
        Type of the generators, among: "solar", "wind", "hydro", "thermal" and "nuclear". Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_pmin: :class:`numpy.ndarray`, dtype:float
        Minimum active power production needed for a generator to work properly. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_pmax: :class:`numpy.ndarray`, dtype:float
        Maximum active power production needed for a generator to work properly. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_redispatchable: :class:`numpy.ndarray`, dtype:bool
        For each generator, it says if the generator is dispatchable or not. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_max_ramp_up: :class:`numpy.ndarray`, dtype:float
        Maximum active power variation possible between two consecutive timestep for each generator:
        a redispatching action
        on generator `g_id` cannot be above :attr:`GridObjects.gen_ramp_up_max` [`g_id`]. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_max_ramp_down: :class:`numpy.ndarray`, dtype:float
        Minimum active power variationpossible between two consecutive timestep for each generator: a redispatching
        action
        on generator `g_id` cannot be below :attr:`GridObjects.gen_ramp_down_min` [`g_id`]. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_min_uptime: :class:`numpy.ndarray`, dtype:float
        The minimum time (expressed in the number of timesteps) a generator needs to be turned on: it's not possible to
        turn off generator `gen_id` that has been turned on less than `gen_min_time_on` [`gen_id`] timesteps
        ago. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_min_downtime: :class:`numpy.ndarray`, dtype:float
        The minimum time (expressed in the number of timesteps) a generator needs to be turned off: it's not possible to
        turn on generator `gen_id` that has been turned off less than `gen_min_time_on` [`gen_id`] timesteps
        ago. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_cost_per_MW: :class:`numpy.ndarray`, dtype:float
        For each generator, it gives the "operating cost", eg the cost, in terms of "used currency" for the production
        of one MW with this generator, if it is already turned on. It's a positive real number. It's the marginal cost
        for each MW. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_startup_cost: :class:`numpy.ndarray`, dtype:float
        The cost to start a generator. It's a positive real number. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_shutdown_cost: :class:`numpy.ndarray`, dtype:float
        The cost to shut down a generator. It's a positive real number. Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    gen_renewable: :class:`numpy.ndarray`, dtype:bool
        Whether each generator is from a renewable energy sources (=can be curtailed). Optional. Used
        for unit commitment problems or redispatching action. [*class attribute*]

    redispatching_unit_commitment_available: ``bool``
        Does the current grid allow for redispatching and / or unit commit problem. If not, any attempt to use it
        will raise a :class:`grid2op.Exceptions.UnitCommitorRedispachingNotAvailable` error. [*class attribute*]
        For an environment to be compatible with this feature, you need to set up, when loading the backend:

          - :attr:`GridObjects.gen_type`
          - :attr:`GridObjects.gen_pmin`
          - :attr:`GridObjects.gen_pmax`
          - :attr:`GridObjects.gen_redispatchable`
          - :attr:`GridObjects.gen_max_ramp_up`
          - :attr:`GridObjects.gen_max_ramp_down`
          - :attr:`GridObjects.gen_min_uptime`
          - :attr:`GridObjects.gen_min_downtime`
          - :attr:`GridObjects.gen_cost_per_MW`
          - :attr:`GridObjects.gen_startup_cost`
          - :attr:`GridObjects.gen_shutdown_cost`
          - :attr:`GridObjects.gen_renewable`

    flexible_load_available: ``bool``
        Does the current grid allow for flexible loads. If not, any attempt to use it
        will raise a :class:`grid2op.Exceptions.UnitCommitorRedispachingNotAvailable` error. [*class attribute*]
        For an environment to be compatible with this feature, you need to set up, when loading the backend:

          - :attr:`GridObjects.load_size`
          - :attr:`GridObjects.load_flexible`
          - :attr:`GridObjects.load_max_ramp_up`
          - :attr:`GridObjects.load_max_ramp_down`
          - :attr:`GridObjects.load_min_uptime`
          - :attr:`GridObjects.load_min_downtime`
          - :attr:`GridObjects.load_cost_per_MW`

    grid_layout: ``dict`` or ``None``
        The layout of the powergrid in a form of a dictionnary with keys the substation name, and value a tuple of
        the coordinate of this substation. If no layout are provided, it defaults to ``None`` [*class attribute*]

    shunts_data_available: ``bool``
        Whether or not the backend support the shunt data. [*class attribute*]

    n_shunt: ``int`` or ``None``
        Number of shunts on the grid. It might be ``None`` if the backend does not support shunts. [*class attribute*]

    name_shunt: ``numpy.ndarray``, dtype:``str`` or ``None``
        Name of each shunt on the grid, or ``None`` if the backend does not support shunts. [*class attribute*]

    shunt_to_subid: :class:`numpy.ndarray`, dtype:int
        for each shunt (if supported), gives the id the substation to which it is connected [*class attribute*]

    storage_type:
        type of each storage units, one of "battery" or "pumped storage"

    storage_Emax:
        maximum energy the storage unit can store, in MWh

    storage_Emin:
        minimum energy in the storage unit, in MWh
        
        At any given point, the state of charge (obs.storage_charge) should be >= than `storage_Emin`. This might
        not be the case if there are losses on your storage units. In this case, the charge can fall below
        this (but the charge will never be < 0.)

    storage_max_p_prod:
        maximum power the storage unit can produce (in MW)

    storage_max_p_absorb :
        maximum power the storage unit can absorb (in MW)

    storage_marginal_cost:
        Cost of usage of the storage unit, when charged or discharged, in $/MWh produced (or absorbed)

    storage_loss:
        The self discharged loss of each storage unit (in MW). It is applicable for each step and each
        storage unit where the state of charge is > 0.
        
        Due to this loss, the storage state of charge can fall below its minimum allowed capacity `storage_Emin`.

    storage_charging_efficiency:
        The efficiency when the storage unit is charging (how much will the capacity increase when the
        unit is charging) between 0. and 1.

    storage_discharging_efficiency:
        The efficiency when the storage unit is discharging (how much will the capacity decrease
        to generate a 1MWh of energy on the grid side) between 0. and 1.

    grid_objects_types: ``matrix``
        Give the information about each element of the "topo_vect" vector. It is an "easy" way to retrieve at
        which element (side of a power, load, generator, storage units) a given component of the "topology vector"
        is referring to.
        For more information, you can consult the :ref:`graph-encoding-gridgraph` of the documentation
        or the getting started notebook about the observation and the action for more information.

    dim_alarms = 0  # TODO
    alarms_area_names = []  # name of each area  # TODO
    alarms_lines_area = {}  # for each lines of the grid, gives on which area(s) it is  # TODO
    alarms_area_lines = []  # for each area in the grid, gives which powerlines it contains # TODO

    dim_alerts: `int`
        The dimension of the "alert space" (number of powerline on which the agent can sent an alert)
        
        .. seealso:: :ref:`grid2op-alert-module` section of the doc for more information
    
        .. versionadded:: 1.9.1
        
    alertable_line_names: `np.ndarray`
        Name (in order) of each powerline on which the agent can send an alarm. It has the size corresponding to :attr:`GridObjects.dim_alerts`
        and contain names of powerlines (string).
        
        .. seealso:: :ref:`grid2op-alert-module` section of the doc for more information
        
        .. versionadded:: 1.9.1
        
    alertable_line_ids: `np.ndarray`
        Id (in order) of each powerline on which the agent can send an alarm. It has the size corresponding to :attr:`GridObjects.dim_alerts`
        and contain ids of powerlines (integer).
        
        .. seealso:: :ref:`grid2op-alert-module` section of the doc for more information

        .. versionadded:: 1.9.1
    """

    BEFORE_COMPAT_VERSION : ClassVar[str] = "neurips_2020_compat"
    glop_version : ClassVar[str] = grid2op.__version__
    
    _INIT_GRID_CLS = None  # do not modify that, this is handled by grid2op automatically
    _PATH_GRID_CLASSES : ClassVar[Optional[str]] = None  # especially do not modify that
    _CLS_DICT : ClassVar[Optional[CLS_AS_DICT_TYPING]] = None  # init once to avoid yet another serialization of the class as dict (in make_cls_dict)
    _CLS_DICT_EXTENDED : ClassVar[Optional[CLS_AS_DICT_TYPING]] = None  # init once to avoid yet another serialization of the class as dict  (in make_cls_dict)

    SUB_COL : ClassVar[int] = 0
    LOA_COL : ClassVar[int] = 1
    GEN_COL : ClassVar[int] = 2
    LOR_COL : ClassVar[int] = 3
    LEX_COL : ClassVar[int] = 4
    STORAGE_COL : ClassVar[int] = 5

    attr_list_vect : ClassVar[Optional[List[str]]] = None
    attr_list_set = {}
    attr_list_json : ClassVar[Optional[List[str]]] = []
    attr_nan_list_set = set()

    # name of the objects
    env_name : ClassVar[str] = "unknown"
    name_load : ClassVar[np.ndarray] = None
    name_gen : ClassVar[np.ndarray] = None
    name_line : ClassVar[np.ndarray] = None
    name_sub : ClassVar[np.ndarray] = None
    name_storage : ClassVar[np.ndarray] = None

    n_busbar_per_sub : ClassVar[int] = DEFAULT_N_BUSBAR_PER_SUB
    n_gen : ClassVar[int] = -1
    n_load : ClassVar[int] = -1
    n_line : ClassVar[int] = -1
    n_sub : ClassVar[int] = -1
    n_storage : ClassVar[int] = -1

    sub_info : ClassVar[np.ndarray] = None
    dim_topo : ClassVar[np.ndarray] = -1

    # to which substation is connected each element
    load_to_subid : ClassVar[np.ndarray] = None
    gen_to_subid : ClassVar[np.ndarray] = None
    line_or_to_subid : ClassVar[np.ndarray] = None
    line_ex_to_subid : ClassVar[np.ndarray] = None
    storage_to_subid : ClassVar[np.ndarray] = None

    # which index has this element in the substation vector
    load_to_sub_pos : ClassVar[np.ndarray] = None
    gen_to_sub_pos : ClassVar[np.ndarray] = None
    line_or_to_sub_pos : ClassVar[np.ndarray] = None
    line_ex_to_sub_pos : ClassVar[np.ndarray] = None
    storage_to_sub_pos : ClassVar[np.ndarray] = None

    # which index has this element in the topology vector
    load_pos_topo_vect : ClassVar[np.ndarray] = None
    gen_pos_topo_vect : ClassVar[np.ndarray] = None
    line_or_pos_topo_vect : ClassVar[np.ndarray] = None
    line_ex_pos_topo_vect : ClassVar[np.ndarray] = None
    storage_pos_topo_vect : ClassVar[np.ndarray] = None

    # "convenient" way to retrieve information of the grid
    grid_objects_types : ClassVar[np.ndarray] = None
    # to which substation each element of the topovect is connected
    _topo_vect_to_sub : ClassVar[np.ndarray] = None

    # list of attribute to convert it from/to a vector
    _vectorized = None

    # for redispatching / unit commitment
    _li_attr_disp : ClassVar[List[str]] = [
        "gen_type",
        "gen_pmin",
        "gen_pmax",
        "gen_redispatchable",
        "gen_max_ramp_up",
        "gen_max_ramp_down",
        "gen_min_uptime",
        "gen_min_downtime",
        "gen_cost_per_MW",
        "gen_startup_cost",
        "gen_shutdown_cost",
        "gen_renewable",
    ]

    _type_attr_disp : ClassVar[List] = [
        str,
        float,
        float,
        bool,
        float,
        float,
        int,
        int,
        float,
        float,
        float,
        bool,
    ]

    # For flexibility / demand response
    _li_attr_flex_load : ClassVar[List[str]] = [
        "load_size",
        "load_flexible",
        "load_max_ramp_up",
        "load_max_ramp_down",
        "load_min_uptime",
        "load_min_downtime",
        "load_cost_per_MW",
    ]

    _type_attr_flex_load : ClassVar[List] = [
        float,
        bool,
        float,
        float,
        int,
        int,
        float
    ]

    # Redispatch data, not available in all Environments
    redispatching_unit_commitment_available : ClassVar[bool] = False
    gen_type : ClassVar[Optional[np.ndarray]] = None
    gen_pmin : ClassVar[Optional[np.ndarray]] = None
    gen_pmax : ClassVar[Optional[np.ndarray]] = None
    gen_redispatchable : ClassVar[Optional[np.ndarray]] = None
    gen_max_ramp_up : ClassVar[Optional[np.ndarray]] = None
    gen_max_ramp_down : ClassVar[Optional[np.ndarray]] = None
    gen_min_uptime : ClassVar[Optional[np.ndarray]] = None
    gen_min_downtime : ClassVar[Optional[np.ndarray]] = None
    gen_cost_per_MW : ClassVar[Optional[np.ndarray]] = None  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )
    gen_startup_cost : ClassVar[Optional[np.ndarray]] = None  # start cost (in currency)
    gen_shutdown_cost : ClassVar[Optional[np.ndarray]] = None  # shutdown cost (in currency)
    gen_renewable : ClassVar[Optional[np.ndarray]] = None

    # Fleixible load data, not available in all Environments
    flexible_load_available: ClassVar[bool] = False
    load_size: ClassVar[Optional[np.ndarray]] = None
    load_flexible: ClassVar[Optional[np.ndarray]] = None
    load_max_ramp_up: ClassVar[Optional[np.ndarray]] = None
    load_max_ramp_down: ClassVar[Optional[np.ndarray]] = None
    load_min_uptime: ClassVar[Optional[np.ndarray]] = None
    load_min_downtime: ClassVar[Optional[np.ndarray]] = None
    load_cost_per_MW: ClassVar[Optional[np.ndarray]] = None

    # Storage unit static data
    storage_type : ClassVar[Optional[np.ndarray]] = None
    storage_Emax : ClassVar[Optional[np.ndarray]] = None
    storage_Emin : ClassVar[Optional[np.ndarray]] = None
    storage_max_p_prod : ClassVar[Optional[np.ndarray]] = None
    storage_max_p_absorb : ClassVar[Optional[np.ndarray]] = None
    storage_marginal_cost : ClassVar[Optional[np.ndarray]] = None
    storage_loss : ClassVar[Optional[np.ndarray]] = None
    storage_charging_efficiency : ClassVar[Optional[np.ndarray]] = None
    storage_discharging_efficiency : ClassVar[Optional[np.ndarray]] = None

    # Grid Layout
    grid_layout : ClassVar[Optional[Dict[str, Tuple[float, float]]]] = None

    # Shunt data, not available in every backend
    shunts_data_available : ClassVar[bool] = False
    n_shunt : ClassVar[Optional[int]] = None
    name_shunt : ClassVar[Optional[np.ndarray]] = None
    shunt_to_subid : ClassVar[Optional[np.ndarray]] = None

    # Alarm / Alert
    assistant_warning_type = None
    
    # Alarm feature
    # dimension of the alarm "space" (number of alarm that can be raised at each step)
    dim_alarms = 0  # TODO
    alarms_area_names = []  # name of each area  # TODO
    alarms_lines_area = (
        {}
    )  # for each lines of the grid, gives on which area(s) it is  # TODO
    alarms_area_lines = (
        []
    )  # for each area in the grid, gives which powerlines it contains # TODO

    # Alert feature 
    # dimension of the alert "space" (number of alerts that can be raised at each step)
    dim_alerts = 0  # TODO
    alertable_line_names = []  # name of each line to produce an alert on # TODO
    alertable_line_ids = []
    
    # test
    _IS_INIT : ClassVar[Optional[bool]] = False
    
    def __init__(self):
        """nothing to do when an object of this class is created, the information is held by the class attributes"""
        pass

    @classmethod
    def set_n_busbar_per_sub(cls, n_busbar_per_sub: N_BUSBAR_PER_SUB_TYPING) -> None:
        # TODO n_busbar_per_sub different num per substations
        cls.n_busbar_per_sub = n_busbar_per_sub
        
    @classmethod
    def tell_dim_alarm(cls, dim_alarms: int) -> None:
        if cls.dim_alarms != 0:
            # number of alarms has already been set, i issue a warning
            warnings.warn(
                "You will change the number of dimensions of the alarm. This might cause trouble "
                "if you environment is read back. We strongly recommend NOT to do this."
            )
        if dim_alarms and cls.assistant_warning_type == "by_line":
            raise Grid2OpException("Impossible to set both alarm and alert for the same environment.")
        
        cls.dim_alarms = dim_alarms
        if dim_alarms:
            cls.assistant_warning_type = "zonal"

    @classmethod
    def tell_dim_alert(cls, dim_alerts: int) -> None:
        if cls.dim_alerts != 0:
            # number of alerts has already been set, i issue a warning
            warnings.warn(
                "You will change the number of dimensions of the alert. This might cause trouble "
                "if you environment is read back. We strongly recommend NOT to do this."
            )
        if dim_alerts and cls.assistant_warning_type == "zonal":
            raise Grid2OpException("Impossible to set both alarm and alert for the same environment.")
        
        cls.dim_alerts = dim_alerts
        if dim_alerts:
            cls.assistant_warning_type = "by_line"

    @classmethod
    def _reset_cls_dict(cls):
        cls._CLS_DICT = None
        cls._CLS_DICT_EXTENDED = None
        
    @classmethod
    def _clear_class_attribute(cls) -> None:
        """Also calls :func:`GridObjects._clear_grid_dependant_class_attributes` : this clear the attribute that
        may be backend dependant too (eg shunts_data)

        This clear the class as if it was defined in grid2op directly.
        """        
        cls.shunts_data_available = False
        cls.n_busbar_per_sub = DEFAULT_N_BUSBAR_PER_SUB
        
        # for redispatching / unit commitment
        cls._li_attr_disp = [
            "gen_type",
            "gen_pmin",
            "gen_pmax",
            "gen_redispatchable",
            "gen_max_ramp_up",
            "gen_max_ramp_down",
            "gen_min_uptime",
            "gen_min_downtime",
            "gen_cost_per_MW",
            "gen_startup_cost",
            "gen_shutdown_cost",
            "gen_renewable",
        ]

        cls._li_attr_flex_load = [
            "load_size",
            "load_flexible",
            "load_max_ramp_up",
            "load_max_ramp_down",
            "load_min_uptime",
            "load_min_downtime",
            "load_cost_per_MW",
        ]

        cls._type_attr_disp = [
            str,
            float,
            float,
            bool,
            float,
            float,
            int,
            int,
            float,
            float,
            float,
            bool,
        ]

        cls._type_attr_flex_load = [
            float,
            bool,
            float,
            float,
            int,
            int,
            float
        ]
        
        cls._clear_grid_dependant_class_attributes()
        
    @classmethod
    def _clear_grid_dependant_class_attributes(cls) -> None:
        """reset to an original state all the class attributes that depends on an environment"""
        cls._reset_cls_dict()
        cls._INIT_GRID_CLS = None  # do not modify that, this is handled by grid2op automatically
        cls._PATH_GRID_CLASSES = None  # especially do not modify that
        
        cls.glop_version = grid2op.__version__

        cls.SUB_COL = 0
        cls.LOA_COL = 1
        cls.GEN_COL = 2
        cls.LOR_COL = 3
        cls.LEX_COL = 4
        cls.STORAGE_COL = 5

        cls.attr_list_vect = None
        cls.attr_list_set = {}
        cls.attr_list_json = []
        cls.attr_nan_list_set = set()

        # class been init
        cls._IS_INIT = False

        # name of the objects
        cls.env_name = "unknown"
        cls.name_load = None
        cls.name_gen = None
        cls.name_line = None
        cls.name_sub = None
        cls.name_storage = None

        cls.n_gen = -1
        cls.n_load = -1
        cls.n_line = -1
        cls.n_sub = -1
        cls.n_storage = -1

        cls.sub_info = None
        cls.dim_topo = -1

        # to which substation is connected each element
        cls.load_to_subid = None
        cls.gen_to_subid = None
        cls.line_or_to_subid = None
        cls.line_ex_to_subid = None
        cls.storage_to_subid = None

        # which index has this element in the substation vector
        cls.load_to_sub_pos = None
        cls.gen_to_sub_pos = None
        cls.line_or_to_sub_pos = None
        cls.line_ex_to_sub_pos = None
        cls.storage_to_sub_pos = None

        # which index has this element in the topology vector
        cls.load_pos_topo_vect = None
        cls.gen_pos_topo_vect = None
        cls.line_or_pos_topo_vect = None
        cls.line_ex_pos_topo_vect = None
        cls.storage_pos_topo_vect = None

        # "convenient" way to retrieve information of the grid
        cls.grid_objects_types = None
        # to which substation each element of the topovect is connected
        cls._topo_vect_to_sub = None

        # list of attribute to convert it from/to a vector
        cls._vectorized = None

        # redispatch data, not available in all environment
        cls.redispatching_unit_commitment_available = False
        cls.gen_type = None
        cls.gen_pmin = None
        cls.gen_pmax = None
        cls.gen_redispatchable = None
        cls.gen_max_ramp_up = None
        cls.gen_max_ramp_down = None
        cls.gen_min_uptime = None
        cls.gen_min_downtime = None
        cls.gen_cost_per_MW = None  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )
        cls.gen_startup_cost = None  # start cost (in currency)
        cls.gen_shutdown_cost = None  # shutdown cost (in currency)
        cls.gen_renewable = None
        # Flexible load data, not available in all environments
        cls.flexible_load_available = False
        cls.load_size = None
        cls.load_flexible = None
        cls.load_max_ramp_up = None
        cls.load_max_ramp_down = None
        cls.load_min_uptime = None
        cls.load_min_downtime = None
        cls.load_cost_per_MW = None

        # storage unit static data
        cls.storage_type = None
        cls.storage_Emax = None
        cls.storage_Emin = None
        cls.storage_max_p_prod = None
        cls.storage_max_p_absorb = None
        cls.storage_marginal_cost = None
        cls.storage_loss = None
        cls.storage_charging_efficiency = None
        cls.storage_discharging_efficiency = None

        # grid layout
        cls.grid_layout = None

        # shunt data, not available in every backend
        cls.n_shunt = None
        cls.name_shunt = None
        cls.shunt_to_subid = None

        # alarm / alert
        cls.assistant_warning_type = None
        
        # alarms
        cls.dim_alarms = 0
        cls.alarms_area_names = []
        cls.alarms_lines_area = {}
        cls.alarms_area_lines = []

        # alerts
        cls.dim_alerts = 0
        cls.alertable_line_names = []
        cls.alertable_line_ids = []
        
    @classmethod
    def _update_value_set(cls) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update the class attribute `attr_list_vect_set` from  `attr_list_vect`
        """
        cls.attr_list_set = set(cls.attr_list_vect)

    def _raise_error_attr_list_none(self) -> None:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Raise a "NotImplementedError" if :attr:`GridObjects.attr_list_vect` is not defined.

        Raises
        -------
        ``NotImplementedError``

        """
        if type(self).attr_list_vect is None:
            raise IncorrectNumberOfElements(
                "attr_list_vect attribute is not defined for class {}. "
                "It is not possible to convert it from/to a vector, "
                "nor to know its size, shape or dtype.".format(type(self))
            )

    def _get_array_from_attr_name(self, attr_name: str) -> Union[np.ndarray, int, str]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This function returns the proper attribute vector that can be inspected in the
        :func:`GridObject.shape`, :func:`GridObject.size`, :func:`GridObject.dtype`,
        :func:`GridObject.from_vect` and :func:`GridObject.to_vect` method.

        If this function is overloaded, then the _assign_attr_from_name must be too.

        Parameters
        ----------
        attr_name: ``str``
            Name of the attribute to inspect or set

        Returns
        -------
        res: ``numpy.ndarray``
            The attribute corresponding the name, flatten as a 1d vector.

        """
        return np.array(getattr(self, attr_name)).flatten()

    def to_vect(self) -> np.ndarray:
        """
        Convert this instance of GridObjects to a numpy ndarray.
        The size of the array is always the same and is determined by the :func:`GridObject.size` method.

        **NB**: in case the class GridObjects is derived,
         either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
         redefined.

        Returns
        -------
        res: ``numpy.ndarray``
            The representation of this action as a flat numpy ndarray

        Examples
        --------
        It is mainly used for converting Observation of Action to vector:

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # for an observation:
            obs = env.reset()
            obs_as_vect = obs.to_vect()

            # for an action
            act = env.action_space.sample()
            ac_as_vect = act.to_vec()

        """

        if self._vectorized is None:
            self._raise_error_attr_list_none()
            li_vect = [
                self._get_array_from_attr_name(el).astype(dt_float)
                for el in type(self).attr_list_vect
            ]
            if li_vect:
                self._vectorized = np.concatenate(li_vect)
            else:
                self._vectorized = np.array([], dtype=dt_float)
        return self._vectorized

    def to_json(self, convert : bool=True) -> Dict[str, Any]:
        """
        Convert this instance of GridObjects to a dictionary that can be json serialized.
        
        .. note::
            This function is different to the :func:`grid2op.Observation.BaseObservation.to_dict`.
            Indeed the dictionnary resulting from this function will count as keys all the attributes
            in :attr:`GridObjects.attr_list_vect` and :attr:`GridObjects.attr_list_json`.
            
            Concretely, if `obs` is an observation (:class:`grid2op.Observation.BaseObservation`)
            then `obs.to_dict()` will have the keys `type(obs).attr_list_vect` and the values will
            be numpy arrays whereas `obs.to_json()` will have the keys
            `type(obs).attr_list_vect` and `type(obs).attr_list_json` and the values will be
            lists (serializable)

        .. warning::
            convert: do you convert the numpy types to standard python list (might take lots of time)
        
        TODO doc and example
        """

        # TODO optimization for action or observation, to reduce json size, for example using the
        # action._modif_inj or action._modif_set_bus etc.
        # for observation this could be using the default values for obs.line_status (always true) etc.
        # or even storing the things in [id, value] for these types of attributes (time_before_cooldown_line,
        # time_before_cooldown_sub, time_next_maintenance, duration_next_maintenance etc.)

        cls = type(self)
        res = {}
        for attr_nm in cls.attr_list_vect + cls.attr_list_json:
            res[attr_nm] = self._get_array_from_attr_name(attr_nm)
        if convert:
            cls._convert_to_json(res)
        return res

    def from_json(self, dict_: Dict[str, Any]) -> None:
        """
        This transform an gridobject (typically an action or an observation) serialized in json format
        to the corresponding grid2op action / observation (subclass of grid2op.Action.BaseAction
        or grid2op.Observation.BaseObservation)

        Parameters
        ----------
        dict_

        Returns
        -------

        """
        # TODO optimization for action or observation, to reduce json size, for example using the see `to_json`
        all_keys = type(self).attr_list_vect + type(self).attr_list_json
        for key, array_ in dict_.items():
            if key not in all_keys:
                raise AmbiguousAction(f'Impossible to recognize the key "{key}"')
            my_attr = getattr(self, key)
            if isinstance(my_attr, np.ndarray):
                # the regular instance is an array, so i just need to assign the right values to it
                my_attr[:] = array_
            else:
                # normal values is a scalar. So i need to convert the array received as a scalar, and
                # convert it to the proper type
                type_ = type(my_attr)
                setattr(self, key, type_(array_[0]))

    @classmethod
    def _convert_to_json(cls, dict_: Dict[str, Any]) -> None:
        for attr_nm in cls.attr_list_vect + cls.attr_list_json:
            tmp = dict_[attr_nm]
            dtype = tmp.dtype
            if dtype == dt_float:
                dict_[attr_nm] = [float(el) for el in tmp]
            elif dtype == dt_int:
                dict_[attr_nm] = [int(el) for el in tmp]
            elif dtype == dt_bool:
                dict_[attr_nm] = [bool(el) for el in tmp]
            elif dtype == float:
                dict_[attr_nm] = [float(el) for el in tmp]
            elif dtype == int:
                dict_[attr_nm] = [int(el) for el in tmp]
            elif dtype == bool:
                dict_[attr_nm] = [bool(el) for el in tmp]

    def shapes(self) -> np.ndarray:
        """
        The shapes of all the components of the action, mainly used for gym compatibility is the shape of all
        part of the action.

        It is mainly used to know of which "sub spaces the action space and observation space are made of, but
        you can also directly use it on an observation or an action.

        It returns a numpy integer array.

        This function must return a vector from which the sum is equal to the return value of "size()".

        The shape vector must have the same number of components as the return value of the :func:`GridObjects.dtype()`
        vector.

        **NB**: in case the class GridObjects is derived,
         either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
         redefined.

        Returns
        -------
        res: ``numpy.ndarray``
            The shape of the :class:`GridObjects`

        Examples
        --------
        It is mainly used to know of which "sub spaces the action space and observation space are made of.

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # for an observation:
            obs_space_shapes = env.observation_space.shape()

            # for an action
            act_space_shapes = env.action_space.shape()

        """
        self._raise_error_attr_list_none()
        res = np.array(
            [self._get_array_from_attr_name(el).shape[0] for el in type(self).attr_list_vect]
        ).astype(dt_int)
        return res

    def dtypes(self) -> np.ndarray:
        """
        The types of the components of the GridObjects, mainly used for gym compatibility is the shape of all part
        of the action.

        It is mainly used to know of which types each "sub spaces" the action space and observation space are made of,
        but you can also directly use it on an observation or an action.

        It is a numpy array of objects.

        The dtype vector must have the same number of components as the return value of the :func:`GridObjects.shape`
        vector.

        **NB**: in case the class GridObjects is derived,
         either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
         redefined.

        Returns
        -------
        res: ``numpy.ndarray``
            The dtype of the :class:`GridObjects`

        Examples
        --------
        It is mainly used to know of which "sub spaces the action space and observation space are made of.

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # for an observation:
            obs_space_types = env.observation_space.dtype()

            # for an action
            act_space_types = env.action_space.dtype()

        """

        self._raise_error_attr_list_none()
        res = np.array(
            [self._get_array_from_attr_name(el).dtype for el in type(self).attr_list_vect]
        )
        return res

    def _assign_attr_from_name(self, attr_nm, vect):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Assign the proper attributes with name 'attr_nm' with the value of the vector vect

        If this function is overloaded, then the _get_array_from_attr_name must be too.

        Parameters
        ----------
        attr_nm
        vect:

        TODO doc : documentation and example
        """
        tmp = getattr(self, attr_nm)
        if isinstance(tmp, (dt_bool, dt_int, dt_float, float, int, bool)):
            if isinstance(vect, np.ndarray):
                setattr(self, attr_nm, vect[0])
            else:
                setattr(self, attr_nm, vect)
        else:
            tmp[:] = vect

    def check_space_legit(self):
        pass

    def from_vect(self, vect, check_legit=True):
        """
        Convert a GridObjects, represented as a vector, into an GridObjects object.

        **NB**: in case the class GridObjects is derived,
        either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
        redefined.

        It is recommended to use it from the action_space and the observation_space exclusively.

        Only the size is checked. If it does not match, an :class:`grid2op.Exceptions.AmbiguousAction` is thrown.
        Otherwise the component of the vector are coerced into the proper type silently.

        It may results in an non deterministic behaviour if the input vector is not a real action, or cannot be
        converted to one.

        Parameters
        ----------
        vect: ``numpy.ndarray``
            A vector representing an BaseAction.

        Examples
        --------
        It is mainly used for converting back vector representing action or observation into "grid2op" action
        or observation. **NB** You should use it only with the "env.action_space" and "env.observation_space"

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # get the vector representation of an observation:
            obs = env.reset()
            obs_as_vect = obs.to_vect()

            # convert it back to an observation (which will be equal to the first one)
            obs_cpy = env.observation_space.from_vect(obs_as_vect)

            # get the vector representation of an action:
            act = env.action_space.sample()
            act_as_vect = act.to_vec()

            # convert it back to an action (which will be equal to the first one)
            act_cpy = env.action_space.from_vect(act_as_vect)

        """
        cls = type(self)
        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements(
                "Incorrect number of elements found while load a GridObjects "
                "from a vector. Found {} elements instead of {}"
                "".format(vect.shape[0], self.size())
            )

        try:
            vect = np.array(vect).astype(dt_float)
        except Exception as exc_:
            raise EnvError(
                "Impossible to convert the input vector to a floating point numpy array "
                "with error:\n"
                '"{}".'.format(exc_)
            )

        self._raise_error_attr_list_none()
        prev_ = 0
        for attr_nm, sh, dt in zip(cls.attr_list_vect, self.shapes(), self.dtypes()):
            tmp = vect[prev_ : (prev_ + sh)]

            # TODO a flag that says "default Nan" for example for when attributes are initialized with
            # nan
            # if np.any(~np.isfinite(tmp)) and default_nan:
            #     raise NonFiniteElement("None finite number in from_vect detected")

            if attr_nm not in cls.attr_nan_list_set and (
                (~np.isfinite(tmp)).any()
            ):
                attrs_debug = []
                prev_debug = 0
                for attr_nm_debug, sh_debug, dt_debug in zip(cls.attr_list_vect, self.shapes(), self.dtypes()):
                    tmp = vect[prev_debug : (prev_debug + sh_debug)]
                    if attr_nm not in cls.attr_nan_list_set and (
                        (~np.isfinite(tmp)).any()):
                        attrs_debug.append(attr_nm_debug)
                    prev_debug += sh_debug
                raise NonFiniteElement(f"None finite number in from_vect "
                                       f"detected for corresponding to attributes "
                                       f"{attrs_debug}")

            try:
                tmp = tmp.astype(dt)
            except Exception as exc_:
                raise EnvError(
                    'Impossible to convert the input vector to its type ({}) for attribute "{}" '
                    "with error:\n"
                    '"{}".'.format(dt, attr_nm, exc_)
                )

            self._assign_attr_from_name(attr_nm, tmp)
            prev_ += sh

        if check_legit:
            self.check_space_legit()

        self._post_process_from_vect()

    def _post_process_from_vect(self):
        """called at the end of "from_vect" if the function requires post processing"""
        pass

    def size(self):
        """
        When the action / observation is converted to a vector, this method return its size.

        NB that it is a requirement that converting an GridObjects gives a vector of a fixed size throughout a training.

        The size of an object if constant, but more: for a given environment the size of each action or the size
        of each observations is constant. This allows us to also define the size of the "action_space" and
        "observation_space": this method also applies to these spaces (see the examples bellow).

        **NB**: in case the class GridObjects is derived,
        either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
        redefined.

        Returns
        -------
        size: ``int``
            The size of the GridObjects if it's converted to a flat vector.

        Examples
        --------
        It is mainly used to know the size of the vector that would represent these objects

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # get the vector representation of an observation:
            obs = env.reset()
            print("The size of this observation is {}".format(obs.size()))

            # get the vector representation of an action:
            act = env.action_space.sample()
            print("The size of this action is {}".format(act.size()))

            # it can also be used with the action_space and observation_space
            print("The size of the observation space is {}".format(env.observation_space.size()))
            print("The size of the action space is {}".format(env.action_space.size()))

        """
        res = self.shapes().sum(dtype=dt_int)
        return res

    @classmethod
    def _aux_pos_big_topo(cls, vect_to_subid, vect_to_sub_pos):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Return the proper "_pos_big_topo" vector given "to_subid" vector and "to_sub_pos" vectors.
        This function is also called to performed sanity check after the load on the powergrid.

        :param vect_to_subid: vector of int giving the id of the topology for this element
        :type vect_to_subid: iterable int

        :param vect_to_sub_pos: vector of int giving the id IN THE SUBSTATION for this element
        :type vect_to_sub_pos: iterable int

        :return:
        """
        res = np.zeros(shape=vect_to_subid.shape, dtype=dt_int)
        for i, (sub_id, my_pos) in enumerate(zip(vect_to_subid, vect_to_sub_pos)):
            obj_before = cls.sub_info[:sub_id].sum()
            res[i] = obj_before + my_pos
        return res

    def _init_class_attr(self, obj=None, _topo_vect_only=False):
        """Init the class attribute from an instance of the class
        
        THIS IS NOT A CLASS ATTR
        
        obj should be an object and NOT a class !
        
        Notes
        -------
        _topo_vect_only: this function is called once when the backend is initialized in `backend.load_grid`  
        (in `backend._compute_pos_big_topo`) and then once when everything is set up 
        (after redispatching and storage data are loaded).
        
        This is why I need the `_topo_vect_only` flag that tells this function when it's called only for 
        `topo_vect` related attributed
        """

        if obj is None:
            obj = self
        cls = type(self)            
        cls_as_dict = {}
        GridObjects._make_cls_dict_extended(obj, cls_as_dict, as_list=False, _topo_vect_only=_topo_vect_only)
        for attr_nm, attr in cls_as_dict.items():
            if _topo_vect_only:
                # safety guard: only set the attribute needed for the computation of the topo_vect vector
                # this should be the only attribute in cls_as_dict but let's be sure 
                if (attr_nm.endswith("to_subid") or
                    attr_nm.endswith("to_sub_pos") or
                    attr_nm.startswith("n_") or
                    attr_nm.startswith("dim_topo") or 
                    attr_nm.startswith("name_") or
                    attr_nm.startswith("shunts_data_available")
                   ):
                    setattr(cls, attr_nm, attr)
            else:
                # set all the attributes
                setattr(cls, attr_nm, attr)
        
        # make sure to catch data intiialized even outside of this function
        if not _topo_vect_only:
            cls._reset_cls_dict()
            tmp = {}
            GridObjects._make_cls_dict_extended(obj, tmp, as_list=False, copy_=True, _topo_vect_only=False)

    def _compute_pos_big_topo(self):
        # move the object attribute as class attribute !
        if not type(self)._IS_INIT:
            self._init_class_attr(_topo_vect_only=True)
        cls = type(self)
        cls._compute_pos_big_topo_cls()
        
    @classmethod
    def _compute_pos_big_topo_cls(cls):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compute the position of each element in the big topological vector.

        Topology action are represented by numpy vector of size np.sum(self.sub_info).
        The vector self.load_pos_topo_vect will give the index of each load in this big topology vector.
        For example, for load i, self.load_pos_topo_vect[i] gives the position in such a topology vector that
        affect this load.

        This position can be automatically deduced from self.sub_info, self.load_to_subid and self.load_to_sub_pos.

        This is the same for generators and both end of powerlines

        :return: ``None``
        """

        # check if we need to implement the position in substation
        if (
            cls.n_storage == -1
            and cls.storage_to_subid is None
            and cls.storage_pos_topo_vect is None
            and cls.storage_to_sub_pos is None
        ):
            # no storage on the grid, so i deactivate them
            cls.set_no_storage()
        cls._compute_sub_elements()  # fill the dim_topo and sub_info attributes
        cls._compute_sub_pos()  # fill the _to_sub_pos attributes
        cls._fill_names()  # fill the name_xxx attributes

        cls.load_pos_topo_vect = cls._aux_pos_big_topo(
            cls.load_to_subid, cls.load_to_sub_pos
        ).astype(dt_int)
        cls.gen_pos_topo_vect = cls._aux_pos_big_topo(
            cls.gen_to_subid, cls.gen_to_sub_pos
        ).astype(dt_int)
        cls.line_or_pos_topo_vect = cls._aux_pos_big_topo(
            cls.line_or_to_subid, cls.line_or_to_sub_pos
        ).astype(dt_int)
        cls.line_ex_pos_topo_vect = cls._aux_pos_big_topo(
            cls.line_ex_to_subid, cls.line_ex_to_sub_pos
        ).astype(dt_int)
        cls.storage_pos_topo_vect = cls._aux_pos_big_topo(
            cls.storage_to_subid, cls.storage_to_sub_pos
        ).astype(dt_int)

        cls._topo_vect_to_sub = np.repeat(np.arange(cls.n_sub), repeats=cls.sub_info)
        cls._check_convert_to_np_array(raise_if_none=False)  # there can still be "None" attribute at this stage
        cls.grid_objects_types = np.full(
            shape=(cls.dim_topo, 6), fill_value=-1, dtype=dt_int
        )
        prev = 0
        for sub_id, nb_el in enumerate(cls.sub_info):
            cls.grid_objects_types[prev : (prev + nb_el), :] = cls.get_obj_substations(
                substation_id=sub_id
            )
            prev += nb_el

    @classmethod
    def _check_sub_id(cls):
        # check it can be converted to proper types
        if not isinstance(cls.load_to_subid, np.ndarray):
            try:
                cls.load_to_subid = np.array(cls.load_to_subid)
                cls.load_to_subid = cls.load_to_subid.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    f"self.load_to_subid should be convertible to a numpy array. "
                    f'It fails with error "{exc_}"'
                )
        if not isinstance(cls.gen_to_subid, np.ndarray):
            try:
                cls.gen_to_subid = np.array(cls.gen_to_subid)
                cls.gen_to_subid = cls.gen_to_subid.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    f"self.gen_to_subid should be convertible to a numpy array. "
                    f'It fails with error "{exc_}"'
                )
        if not isinstance(cls.line_or_to_subid, np.ndarray):
            try:
                cls.line_or_to_subid = np.array(cls.line_or_to_subid)
                cls.line_or_to_subid = cls.line_or_to_subid.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    f"self.line_or_to_subid should be convertible to a numpy array. "
                    f'It fails with error "{exc_}"'
                )
        if not isinstance(cls.line_ex_to_subid, np.ndarray):
            try:
                cls.line_ex_to_subid = np.array(cls.line_ex_to_subid)
                cls.line_ex_to_subid = cls.line_ex_to_subid.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.line_ex_to_subid should be convertible to a numpy array"
                    f'It fails with error "{exc_}"'
                )

        if not isinstance(cls.storage_to_subid, np.ndarray):
            try:
                cls.storage_to_subid = np.array(cls.storage_to_subid)
                cls.storage_to_subid = cls.storage_to_subid.astype(dt_int)
            except Exception as e:
                raise EnvError(
                    "self.storage_to_subid should be convertible to a numpy array"
                )

        # now check the sizes
        if len(cls.load_to_subid) != cls.n_load:
            raise IncorrectNumberOfLoads()
        if np.min(cls.load_to_subid) < 0:
            raise EnvError("Some shunt is connected to a negative substation id.")
        if np.max(cls.load_to_subid) > cls.n_sub:
            raise EnvError(
                "Some load is supposed to be connected to substations with id {} which"
                "is greater than the number of substations of the grid, which is {}."
                "".format(np.max(cls.load_to_subid), cls.n_sub)
            )

        if len(cls.gen_to_subid) != cls.n_gen:
            raise IncorrectNumberOfGenerators()
        if np.min(cls.gen_to_subid) < 0:
            raise EnvError("Some shunt is connected to a negative substation id.")
        if np.max(cls.gen_to_subid) > cls.n_sub:
            raise EnvError(
                "Some generator is supposed to be connected to substations with id {} which"
                "is greater than the number of substations of the grid, which is {}."
                "".format(np.max(cls.gen_to_subid), cls.n_sub)
            )
        if len(cls.line_or_to_subid) != cls.n_line:
            raise IncorrectNumberOfLines()
        if np.min(cls.line_or_to_subid) < 0:
            raise EnvError("Some shunt is connected to a negative substation id.")
        if np.max(cls.line_or_to_subid) > cls.n_sub:
            raise EnvError(
                "Some powerline (or) is supposed to be connected to substations with id {} which"
                "is greater than the number of substations of the grid, which is {}."
                "".format(np.max(cls.line_or_to_subid), cls.n_sub)
            )

        if len(cls.line_ex_to_subid) != cls.n_line:
            raise IncorrectNumberOfLines()
        if np.min(cls.line_ex_to_subid) < 0:
            raise EnvError("Some shunt is connected to a negative substation id.")
        if np.max(cls.line_ex_to_subid) > cls.n_sub:
            raise EnvError(
                "Some powerline (ex) is supposed to be connected to substations with id {} which"
                "is greater than the number of substations of the grid, which is {}."
                "".format(np.max(cls.line_or_to_subid), cls.n_sub)
            )
        if len(cls.storage_to_subid) != cls.n_storage:
            raise IncorrectNumberOfStorages()

        if cls.n_storage > 0:
            if np.min(cls.storage_to_subid) < 0:
                raise EnvError("Some storage is connected to a negative substation id.")
            if np.max(cls.storage_to_subid) > cls.n_sub:
                raise EnvError(
                    "Some powerline (ex) is supposed to be connected to substations with id {} which"
                    "is greater than the number of substations of the grid, which is {}."
                    "".format(np.max(cls.line_or_to_subid), cls.n_sub)
                )
            
    @classmethod
    def _fill_names(cls):
        """fill the name vectors (**eg** name_line) if not done already in the backend.
        This function is used to fill the name of the class.
        """
        if cls.name_line is None:
            cls.name_line = [
                "{}_{}_{}".format(or_id, ex_id, l_id)
                for l_id, (or_id, ex_id) in enumerate(
                    zip(cls.line_or_to_subid, cls.line_ex_to_subid)
                )
            ]
            cls.name_line = np.array(cls.name_line)
            warnings.warn(
                "name_line is None so default line names have been assigned to your grid. "
                "(FYI: Line names are used to make the correspondence between the chronics and the backend)"
                "This might result in impossibility to load data."
                '\n\tIf "env.make" properly worked, you can safely ignore this warning.'
            )
            cls._reset_cls_dict()
            
        if cls.name_load is None:
            cls.name_load = [
                "load_{}_{}".format(bus_id, load_id)
                for load_id, bus_id in enumerate(cls.load_to_subid)
            ]
            cls.name_load = np.array(cls.name_load)
            warnings.warn(
                "name_load is None so default load names have been assigned to your grid. "
                "(FYI: load names are used to make the correspondence between the chronics and the backend)"
                "This might result in impossibility to load data."
                '\n\tIf "env.make" properly worked, you can safely ignore this warning.'
            )
            cls._reset_cls_dict()
            
        if cls.name_gen is None:
            cls.name_gen = [
                "gen_{}_{}".format(bus_id, gen_id)
                for gen_id, bus_id in enumerate(cls.gen_to_subid)
            ]
            cls.name_gen = np.array(cls.name_gen)
            warnings.warn(
                "name_gen is None so default generator names have been assigned to your grid. "
                "(FYI: generator names are used to make the correspondence between the chronics and "
                "the backend)"
                "This might result in impossibility to load data."
                '\n\tIf "env.make" properly worked, you can safely ignore this warning.'
            )
            cls._reset_cls_dict()
            
        if cls.name_sub is None:
            cls.name_sub = ["sub_{}".format(sub_id) for sub_id in range(cls.n_sub)]
            cls.name_sub = np.array(cls.name_sub)
            warnings.warn(
                "name_sub is None so default substation names have been assigned to your grid. "
                "(FYI: substation names are used to make the correspondence between the chronics and "
                "the backend)"
                "This might result in impossibility to load data."
                '\n\tIf "env.make" properly worked, you can safely ignore this warning.'
            )
            cls._reset_cls_dict()
            
        if cls.name_storage is None:
            cls.name_storage = [
                "storage_{}_{}".format(bus_id, sto_id)
                for sto_id, bus_id in enumerate(cls.storage_to_subid)
            ]
            cls.name_storage = np.array(cls.name_storage)
            warnings.warn(
                "name_storage is None so default storage unit names have been assigned to your grid. "
                "(FYI: storage names are used to make the correspondence between the chronics and "
                "the backend)"
                "This might result in impossibility to load data."
                '\n\tIf "env.make" properly worked, you can safely ignore this warning.'
            )
            cls._reset_cls_dict()
            
        if cls.shunts_data_available and cls.name_shunt is None:
            if cls.shunt_to_subid is not None:
                # used for legacy lightsim2grid
                # shunt names were defined after...
                cls.name_shunt = [
                    "shunt_{}_{}".format(bus_id, sh_id)
                    for sh_id, bus_id in enumerate(cls.shunt_to_subid)
                ]
                cls.name_shunt = np.array(cls.name_shunt)
                warnings.warn(
                    "name_shunt is None so default shunt names have been assigned to your grid. "
                    "(FYI: shunt names are used to make the correspondence between the chronics and "
                    "the backend)"
                    "This might result in impossibility to load data."
                    '\n\tIf "env.make" properly worked, you can safely ignore this warning.'
                )
                cls._reset_cls_dict()

    @classmethod
    def _check_names(cls):
        cls._fill_names()

        if not isinstance(cls.name_line, np.ndarray):
            try:
                cls.name_line = np.array(cls.name_line)
                cls.name_line = cls.name_line.astype(str)
            except Exception as exc_:
                raise EnvError(
                    f"self.name_line should be convertible to a numpy array of type str"
                ) from exc_
        if not isinstance(cls.name_load, np.ndarray):
            try:
                cls.name_load = np.array(cls.name_load)
                cls.name_load = cls.name_load.astype(str)
            except Exception as exc_:
                raise EnvError(
                    "self.name_load should be convertible to a numpy array of type str."
                ) from exc_
        if not isinstance(cls.name_gen, np.ndarray):
            try:
                cls.name_gen = np.array(cls.name_gen)
                cls.name_gen = cls.name_gen.astype(str)
            except Exception as exc_:
                raise EnvError(
                    "self.name_gen should be convertible to a numpy array of type str."
                ) from exc_
        if not isinstance(cls.name_sub, np.ndarray):
            try:
                cls.name_sub = np.array(cls.name_sub)
                cls.name_sub = cls.name_sub.astype(str)
            except Exception as exc_:
                raise EnvError(
                    "self.name_sub should be convertible to a numpy array of type str."
                ) from exc_
        if not isinstance(cls.name_storage, np.ndarray):
            try:
                cls.name_storage = np.array(cls.name_storage)
                cls.name_storage = cls.name_storage.astype(str)
            except Exception as exc_:
                raise EnvError(
                    "self.name_storage should be convertible to a numpy array of type str."
                ) from exc_

        attrs_nms = [
            cls.name_gen,
            cls.name_sub,
            cls.name_line,
            cls.name_load,
            cls.name_storage,
        ]
        nms = ["generators", "substations", "lines", "loads", "storage units"]
        if cls.shunts_data_available:
            # these are set to "None" if there is no shunts on the grid
            attrs_nms.append(cls.name_shunt)
            nms.append("shunts")

        for arr_, nm in zip(attrs_nms, nms):
            try:
                tmp = np.unique(arr_)
                tmp.shape[0]
                arr_.shape[0]
            except AttributeError as exc_:
                raise Grid2OpException(f"Error for {nm}: name is most likely None") from exc_
            
            if tmp.shape[0] != arr_.shape[0]:
                nms = "\n\t - ".join(sorted(arr_))
                raise EnvError(
                    f'Two {nm} have the same names. Please check the "grid.json" file and make sure the '
                    f"name of the {nm} are all different. Right now they are \n\t - {nms}."
                )

    @classmethod
    def _check_sub_pos(cls):
        if not isinstance(cls.load_to_sub_pos, np.ndarray):
            try:
                cls.load_to_sub_pos = np.array(cls.load_to_sub_pos)
                cls.load_to_sub_pos = cls.load_to_sub_pos.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.load_to_sub_pos should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.gen_to_sub_pos, np.ndarray):
            try:
                cls.gen_to_sub_pos = np.array(cls.gen_to_sub_pos)
                cls.gen_to_sub_pos = cls.gen_to_sub_pos.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.gen_to_sub_pos should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.line_or_to_sub_pos, np.ndarray):
            try:
                cls.line_or_to_sub_pos = np.array(cls.line_or_to_sub_pos)
                cls.line_or_to_sub_pos = cls.line_or_to_sub_pos.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.line_or_to_sub_pos should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.line_ex_to_sub_pos, np.ndarray):
            try:
                cls.line_ex_to_sub_pos = np.array(cls.line_ex_to_sub_pos)
                cls.line_ex_to_sub_pos = cls.line_ex_to_sub_pos.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.line_ex_to_sub_pos should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.storage_to_sub_pos, np.ndarray):
            try:
                cls.storage_to_sub_pos = np.array(cls.storage_to_sub_pos)
                cls.storage_to_sub_pos = cls.storage_to_sub_pos.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.line_ex_to_sub_pos should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )

    @classmethod
    def _check_topo_vect(cls):
        if not isinstance(cls.load_pos_topo_vect, np.ndarray):
            try:
                cls.load_pos_topo_vect = np.array(cls.load_pos_topo_vect)
                cls.load_pos_topo_vect = cls.load_pos_topo_vect.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.load_pos_topo_vect should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.gen_pos_topo_vect, np.ndarray):
            try:
                cls.gen_pos_topo_vect = np.array(cls.gen_pos_topo_vect)
                cls.gen_pos_topo_vect = cls.gen_pos_topo_vect.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.gen_pos_topo_vect should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.line_or_pos_topo_vect, np.ndarray):
            try:
                cls.line_or_pos_topo_vect = np.array(cls.line_or_pos_topo_vect)
                cls.line_or_pos_topo_vect = cls.line_or_pos_topo_vect.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.line_or_pos_topo_vect should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.line_ex_pos_topo_vect, np.ndarray):
            try:
                cls.line_ex_pos_topo_vect = np.array(cls.line_ex_pos_topo_vect)
                cls.line_ex_pos_topo_vect = cls.line_ex_pos_topo_vect.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.line_ex_pos_topo_vect should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )
        if not isinstance(cls.storage_pos_topo_vect, np.ndarray):
            try:
                cls.storage_pos_topo_vect = np.array(cls.storage_pos_topo_vect)
                cls.storage_pos_topo_vect = cls.storage_pos_topo_vect.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    "self.storage_pos_topo_vect should be convertible to a numpy array. Error was "
                    f"{exc_}"
                )

    @classmethod
    def _compute_sub_pos(cls):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is used at the initialization of the environment.

        Export to grid2op the position of each object in their substation
        If not done by the user, we will order the objects the following way, for each substation:

        - load (if any is connected to this substation) will be labeled first
        - gen will be labeled just after
        - then origin side of powerline
        - then extremity side of powerline

        you are free to chose any other ordering. It's a possible ordering we propose for the example, but it is
        definitely not mandatory.

        It supposes that the *_to_sub_id are properly set up
        """

        need_implement = False
        if cls.load_to_sub_pos is None:
            need_implement = True
        if cls.gen_to_sub_pos is None:
            if need_implement is False:
                raise BackendError(
                    'You chose not to implement "gen_to_sub_pos" but not "load_to_sub_pos". We cannot '
                    "work with that. Please either use the automatic setting, or implement all of "
                    "*_to_sub_pos vectors"
                    ""
                )
            need_implement = True
        if cls.line_or_to_sub_pos is None:
            if need_implement is False:
                raise BackendError(
                    'You chose not to implement "line_or_to_sub_pos" but "load_to_sub_pos"'
                    'or "gen_to_sub_pos". We cannot '
                    "work with that. Please either use the automatic setting, or implement all of "
                    "*_to_sub_pos vectors"
                    ""
                )
            need_implement = True
        if cls.line_ex_to_sub_pos is None:
            if need_implement is False:
                raise BackendError(
                    'You chose not to implement "line_ex_to_sub_pos" but "load_to_sub_pos"'
                    'or "gen_to_sub_pos" or "line_or_to_sub_pos". We cannot '
                    "work with that. Please either use the automatic setting, or implement all of "
                    "*_to_sub_pos vectors"
                    ""
                )
            need_implement = True
        if cls.storage_to_sub_pos is None:
            if need_implement is False:
                raise BackendError(
                    'You chose not to implement "storage_to_sub_pos" but "load_to_sub_pos"'
                    'or "gen_to_sub_pos" or "line_or_to_sub_pos" or "line_ex_to_sub_pos". '
                    "We cannot "
                    "work with that. Please either use the automatic setting, or implement all of "
                    "*_to_sub_pos vectors"
                    ""
                )
            need_implement = True

        if not need_implement:
            return

        last_order_number = np.zeros(cls.n_sub, dtype=dt_int)
        cls.load_to_sub_pos = np.zeros(cls.n_load, dtype=dt_int)
        for load_id, sub_id_connected in enumerate(cls.load_to_subid):
            cls.load_to_sub_pos[load_id] = last_order_number[sub_id_connected]
            last_order_number[sub_id_connected] += 1

        cls.gen_to_sub_pos = np.zeros(cls.n_gen, dtype=dt_int)
        for gen_id, sub_id_connected in enumerate(cls.gen_to_subid):
            cls.gen_to_sub_pos[gen_id] = last_order_number[sub_id_connected]
            last_order_number[sub_id_connected] += 1

        cls.line_or_to_sub_pos = np.zeros(cls.n_line, dtype=dt_int)
        for lor_id, sub_id_connected in enumerate(cls.line_or_to_subid):
            cls.line_or_to_sub_pos[lor_id] = last_order_number[sub_id_connected]
            last_order_number[sub_id_connected] += 1

        cls.line_ex_to_sub_pos = np.zeros(cls.n_line, dtype=dt_int)
        for lex_id, sub_id_connected in enumerate(cls.line_ex_to_subid):
            cls.line_ex_to_sub_pos[lex_id] = last_order_number[sub_id_connected]
            last_order_number[sub_id_connected] += 1

        cls.storage_to_sub_pos = np.zeros(cls.n_storage, dtype=dt_int)
        for sto_id, sub_id_connected in enumerate(cls.storage_to_subid):
            cls.storage_to_sub_pos[sto_id] = last_order_number[sub_id_connected]
            last_order_number[sub_id_connected] += 1

    @classmethod
    def _compute_sub_elements(cls):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\


        Computes "dim_topo" and "sub_info" class attributes

        It supposes that *to_subid are initialized and that n_line, n_sub, n_load and n_gen are all positive
        """
        if cls.dim_topo is None or cls.dim_topo <= 0:
            cls.dim_topo = 2 * cls.n_line + cls.n_load + cls.n_gen + cls.n_storage

        if cls.sub_info is None:
            cls.sub_info = np.zeros(cls.n_sub, dtype=dt_int)
            # NB the vectorized implementation do not work
            for s_id in cls.load_to_subid:
                cls.sub_info[s_id] += 1
            for s_id in cls.gen_to_subid:
                cls.sub_info[s_id] += 1
            for s_id in cls.line_or_to_subid:
                cls.sub_info[s_id] += 1
            for s_id in cls.line_ex_to_subid:
                cls.sub_info[s_id] += 1
            for s_id in cls.storage_to_subid:
                cls.sub_info[s_id] += 1

    @classmethod
    def _assign_attr(cls, attrs_list, tp, tp_nm, raise_if_none=False):
        for el in attrs_list:
            arr = getattr(cls, el)
            if arr is None:
                if raise_if_none:
                    raise Grid2OpException(f"class attribute {el} is None, but should not be.")
                continue
            try:
                arr2 = np.array(arr).astype(tp)
            except ValueError as exc_:
                raise Grid2OpException(f"Impossible to convert attribute name {el} to {tp_nm} for attr {el}") from exc_
            if len(arr) != len(arr2):
                raise Grid2OpException(f"During the conversion to {tp} for attr {el} an error occured (results have not the proper size {len(arr2)} vs {len(arr)})")
            if (arr != arr2).any():
                mask = arr != arr2
                raise Grid2OpException(f"Impossible to safely convert attribute name {el} to {tp_nm} for attr {el}: {arr[mask]} vs {arr2[mask]}.")
            setattr(cls, el, arr2)
        
    @classmethod
    def _check_convert_to_np_array(cls, raise_if_none=True):
        # convert int to array of ints
        attrs_int = ["load_pos_topo_vect",
                     "load_to_subid",
                     "load_to_sub_pos",
                     "gen_pos_topo_vect",
                     "gen_to_subid",
                     "gen_to_sub_pos",
                     "storage_pos_topo_vect",
                     "storage_to_subid",
                     "storage_to_sub_pos",
                     "line_or_pos_topo_vect",
                     "line_or_to_subid",
                     "line_or_to_sub_pos",
                     "line_ex_pos_topo_vect",
                     "line_ex_to_subid",
                     "line_ex_to_sub_pos"]
        if cls.redispatching_unit_commitment_available:
            attrs_int.append("gen_min_uptime")
            attrs_int.append("gen_min_downtime")
        if cls.flexible_load_available:
            attrs_int.append("load_min_uptime")
            attrs_int.append("load_min_downtime")
            
        cls._assign_attr(attrs_int, dt_int, "int", raise_if_none)
        
        # convert str to array of str
        attrs_str = ["name_load",
                     "name_gen",
                     "name_line", 
                     "name_sub", 
                     "name_storage",
                     "storage_type"]
        if cls.redispatching_unit_commitment_available:
            attrs_str.append("gen_type")
        cls._assign_attr(attrs_str, str, "str", raise_if_none)
        
        # convert float to array of float
        attrs_float = ["storage_Emax",
                       "storage_Emin",
                       "storage_max_p_prod",
                       "storage_max_p_absorb",
                       "storage_marginal_cost",
                       "storage_loss",
                       "storage_charging_efficiency",
                       "storage_discharging_efficiency"]
        if cls.redispatching_unit_commitment_available:
            attrs_float += ["gen_pmin",
                            "gen_pmax",
                            "gen_max_ramp_up",
                            "gen_max_ramp_down",
                            "gen_cost_per_MW",
                            "gen_startup_cost",
                            "gen_shutdown_cost"]
        if cls.flexible_load_available:
            attrs_float += ["load_size",
                            "load_max_ramp_up",
                            "load_max_ramp_down",
                            "load_cost_per_MW"]
        cls._assign_attr(attrs_float, dt_float, "float", raise_if_none)
    
    @classmethod
    def assert_grid_correct_cls(cls):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is used at the initialization of the environment.

        Performs some checking on the loaded grid to make sure it is consistent.

        It also makes sure that the vector such as *sub_info*, *load_to_subid* or *gen_to_sub_pos* are of the
        right type eg. numpy.ndarray with dtype: dt_int

        It is called after the grid has been loaded.

        These function is by default called by the :class:`grid2op.Environment` class after the initialization of the
        environment.
        If these tests are not successfull, no guarantee are given that the backend will return consistent computations.

        In order for the backend to fully understand the structure of actions, it is strongly advised NOT to override
        this method.

        :return: ``None``
        :raise: :class:`grid2op.EnvError` and possibly all of its derived class.
        """
        # TODO refactor this method with the `_check***` methods.
        # TODO refactor the `_check***` to use the same "base functions" that would be coded only once.

        # TODO n_busbar_per_sub different num per substations
        if isinstance(cls.n_busbar_per_sub, (int, dt_int, np.int32, np.int64)):
            cls.n_busbar_per_sub = dt_int(cls.n_busbar_per_sub)
                                   # np.full(cls.n_sub,
                                   #         fill_value=cls.n_busbar_per_sub,
                                   #         dtype=dt_int)
        else:
            # cls.n_busbar_per_sub = np.array(cls.n_busbar_per_sub)
            # cls.n_busbar_per_sub = cls.n_busbar_per_sub.astype(dt_int)
            raise EnvError("Grid2op cannot handle a different number of busbar per substations at the moment.")
        
        # if cls.n_busbar_per_sub != int(cls.n_busbar_per_sub):
            # raise EnvError(f"`n_busbar_per_sub` should be convertible to an integer, found {cls.n_busbar_per_sub}")
        # cls.n_busbar_per_sub = int(cls.n_busbar_per_sub)
        if (cls.n_busbar_per_sub < 1).any():
            raise EnvError(f"`n_busbar_per_sub` should be >= 1 found {cls.n_busbar_per_sub}")
            
        if cls.n_gen <= 0:
            raise EnvError(
                "n_gen is negative. Powergrid is invalid: there are no generator"
            )
        if cls.n_load <= 0:
            raise EnvError(
                "n_load is negative. Powergrid is invalid: there are no load"
            )
        if cls.n_line <= 0:
            raise EnvError(
                "n_line is negative. Powergrid is invalid: there are no line"
            )
        if cls.n_sub <= 0:
            raise EnvError(
                "n_sub is negative. Powergrid is invalid: there are no substation"
            )

        if (
            cls.n_storage == -1
            and cls.storage_to_subid is None
            and cls.storage_pos_topo_vect is None
            and cls.storage_to_sub_pos is None
        ):
            # no storage on the grid, so i deactivate them
            cls.set_no_storage()

        if cls.n_storage < 0:
            raise EnvError(
                "n_storage is negative. Powergrid is invalid: you specify a negative number of unit storage"
            )

        cls._compute_sub_elements()
        if not isinstance(cls.sub_info, np.ndarray):
            try:
                cls.sub_info = np.array(cls.sub_info)
                cls.sub_info = cls.sub_info.astype(dt_int)
            except Exception as exc_:
                raise EnvError(
                    f"self.sub_info should be convertible to a numpy array. "
                    f'It fails with error "{exc_}"'
                )
        # check everything can be converted to numpy array of right types
        cls._check_convert_to_np_array()
        
        # to which subtation they are connected
        cls._check_sub_id()

        # compute the position in substation if not done already
        cls._compute_sub_pos()

        # test position in substation
        cls._check_sub_pos()

        # test position in topology vector
        cls._check_topo_vect()

        # test that all numbers are finite:
        tmp = np.concatenate(
            (
                cls.sub_info.flatten(),
                cls.load_to_subid.flatten(),
                cls.gen_to_subid.flatten(),
                cls.line_or_to_subid.flatten(),
                cls.line_ex_to_subid.flatten(),
                cls.storage_to_subid.flatten(),
                cls.load_to_sub_pos.flatten(),
                cls.gen_to_sub_pos.flatten(),
                cls.line_or_to_sub_pos.flatten(),
                cls.line_ex_to_sub_pos.flatten(),
                cls.storage_to_sub_pos.flatten(),
                cls.load_pos_topo_vect.flatten(),
                cls.gen_pos_topo_vect.flatten(),
                cls.line_or_pos_topo_vect.flatten(),
                cls.line_ex_pos_topo_vect.flatten(),
                cls.storage_pos_topo_vect.flatten(),
            )
        )
        try:
            if (~np.isfinite(tmp)).any():
                raise EnvError(
                    "The grid could not be loaded properly."
                    "One of the vector is made of non finite elements, check the sub_info, *_to_subid, "
                    "*_to_sub_pos and *_pos_topo_vect vectors"
                )
        except Exception as exc_:
            raise EnvError(
                f"Impossible to check whether or not vectors contains only finite elements (probably one "
                f"or more topology related vector is not valid (contains ``None``). Error was "
                f"{exc_}"
            )

        # check sizes
        if len(cls.sub_info) != cls.n_sub:
            raise IncorrectNumberOfSubstation(
                "The number of substation is not consistent in "
                'self.sub_info (size "{}")'
                "and  self.n_sub ({})".format(len(cls.sub_info), cls.n_sub)
            )
        if (
            cls.sub_info.sum()
            != cls.n_load + cls.n_gen + 2 * cls.n_line + cls.n_storage
        ):
            err_msg = "The number of elements of elements is not consistent between self.sub_info where there are "
            err_msg += (
                "{} elements connected to all substations and the number of load, generators and lines in "
                "the _grid ({})."
            )
            err_msg = err_msg.format(
                cls.sub_info.sum(),
                cls.n_load + cls.n_gen + 2 * cls.n_line + cls.n_storage,
            )
            raise IncorrectNumberOfElements(err_msg)


        # for names
        cls._check_names()
        
        if len(cls.name_load) != cls.n_load:
            raise IncorrectNumberOfLoads("len(self.name_load) != self.n_load")
        if len(cls.name_gen) != cls.n_gen:
            raise IncorrectNumberOfGenerators("len(self.name_gen) != self.n_gen")
        if len(cls.name_line) != cls.n_line:
            raise IncorrectNumberOfLines("len(self.name_line) != self.n_line")
        if len(cls.name_storage) != cls.n_storage:
            raise IncorrectNumberOfStorages("len(self.name_storage) != self.n_storage")
        if len(cls.name_sub) != cls.n_sub:
            raise IncorrectNumberOfSubstation("len(self.name_sub) != self.n_sub")

        if len(cls.load_to_sub_pos) != cls.n_load:
            raise IncorrectNumberOfLoads("len(self.load_to_sub_pos) != self.n_load")
        if len(cls.gen_to_sub_pos) != cls.n_gen:
            raise IncorrectNumberOfGenerators("en(self.gen_to_sub_pos) != self.n_gen")
        if len(cls.line_or_to_sub_pos) != cls.n_line:
            raise IncorrectNumberOfLines("len(self.line_or_to_sub_pos) != self.n_line")
        if len(cls.line_ex_to_sub_pos) != cls.n_line:
            raise IncorrectNumberOfLines("len(self.line_ex_to_sub_pos) != self.n_line")
        if len(cls.storage_to_sub_pos) != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "len(self.storage_to_sub_pos) != self.n_storage"
            )

        if len(cls.load_pos_topo_vect) != cls.n_load:
            raise IncorrectNumberOfLoads("len(self.load_pos_topo_vect) != self.n_load")
        if len(cls.gen_pos_topo_vect) != cls.n_gen:
            raise IncorrectNumberOfGenerators(
                "len(self.gen_pos_topo_vect) != self.n_gen"
            )
        if len(cls.line_or_pos_topo_vect) != cls.n_line:
            raise IncorrectNumberOfLines(
                "len(self.line_or_pos_topo_vect) != self.n_line"
            )
        if len(cls.line_ex_pos_topo_vect) != cls.n_line:
            raise IncorrectNumberOfLines(
                "len(self.line_ex_pos_topo_vect) != self.n_line"
            )
        if len(cls.storage_pos_topo_vect) != cls.n_storage:
            raise IncorrectNumberOfLines(
                "len(self.storage_pos_topo_vect) != self.n_storage"
            )

        # test if object are connected to right substation
        obj_per_sub = np.zeros(shape=(cls.n_sub,), dtype=dt_int)
        for sub_id in cls.load_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in cls.gen_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in cls.line_or_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in cls.line_ex_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in cls.storage_to_subid:
            obj_per_sub[sub_id] += 1

        if not np.all(obj_per_sub == cls.sub_info):
            raise IncorrectNumberOfElements(
                f"for substation(s): {(obj_per_sub != cls.sub_info).nonzero()[0]}"
            )

        # test right number of element in substations
        # test that for each substation i don't have an id above the number of element of a substations
        for i, (sub_id, sub_pos) in enumerate(
            zip(cls.load_to_subid, cls.load_to_sub_pos)
        ):
            if sub_pos >= cls.sub_info[sub_id]:
                raise IncorrectPositionOfLoads("for load {}".format(i))
        for i, (sub_id, sub_pos) in enumerate(
            zip(cls.gen_to_subid, cls.gen_to_sub_pos)
        ):
            if sub_pos >= cls.sub_info[sub_id]:
                raise IncorrectPositionOfGenerators("for generator {}".format(i))
        for i, (sub_id, sub_pos) in enumerate(
            zip(cls.line_or_to_subid, cls.line_or_to_sub_pos)
        ):
            if sub_pos >= cls.sub_info[sub_id]:
                raise IncorrectPositionOfLines("for line {} at origin side".format(i))
        for i, (sub_id, sub_pos) in enumerate(
            zip(cls.line_ex_to_subid, cls.line_ex_to_sub_pos)
        ):
            if sub_pos >= cls.sub_info[sub_id]:
                raise IncorrectPositionOfLines("for line {} at extremity side".format(i))
        for i, (sub_id, sub_pos) in enumerate(
            zip(cls.storage_to_subid, cls.storage_to_sub_pos)
        ):
            if sub_pos >= cls.sub_info[sub_id]:
                raise IncorrectPositionOfStorages("for storage {}".format(i))

        # check that i don't have 2 objects with the same id in the "big topo" vector
        concat_topo = np.concatenate(
            (
                cls.load_pos_topo_vect.flatten(),
                cls.gen_pos_topo_vect.flatten(),
                cls.line_or_pos_topo_vect.flatten(),
                cls.line_ex_pos_topo_vect.flatten(),
                cls.storage_pos_topo_vect.flatten(),
            )
        )
        if len(np.unique(concat_topo)) !=cls.sub_info.sum():
            raise EnvError(
                "2 different objects would have the same id in the topology vector, or there would be"
                "an empty component in this vector."
            )

        # check that self.load_pos_topo_vect and co are consistent
        load_pos_big_topo = cls._aux_pos_big_topo(
            cls.load_to_subid, cls.load_to_sub_pos
        )
        if not np.all(load_pos_big_topo == cls.load_pos_topo_vect):
            raise IncorrectPositionOfLoads(
                "Mismatch between load_to_subid, load_to_sub_pos and load_pos_topo_vect"
            )
        gen_pos_big_topo = cls._aux_pos_big_topo(cls.gen_to_subid, cls.gen_to_sub_pos)
        if not np.all(gen_pos_big_topo == cls.gen_pos_topo_vect):
            raise IncorrectNumberOfGenerators(
                "Mismatch between gen_to_subid, gen_to_sub_pos and gen_pos_topo_vect"
            )
        lines_or_pos_big_topo = cls._aux_pos_big_topo(
            cls.line_or_to_subid, cls.line_or_to_sub_pos
        )
        if not np.all(lines_or_pos_big_topo == cls.line_or_pos_topo_vect):
            raise IncorrectPositionOfLines(
                "Mismatch between line_or_to_subid, "
                "line_or_to_sub_pos and line_or_pos_topo_vect"
            )
        lines_ex_pos_big_topo = cls._aux_pos_big_topo(
            cls.line_ex_to_subid, cls.line_ex_to_sub_pos
        )
        if not np.all(lines_ex_pos_big_topo == cls.line_ex_pos_topo_vect):
            raise IncorrectPositionOfLines(
                "Mismatch between line_ex_to_subid, "
                "line_ex_to_sub_pos and line_ex_pos_topo_vect"
            )
        storage_pos_big_topo = cls._aux_pos_big_topo(
            cls.storage_to_subid, cls.storage_to_sub_pos
        )
        if not np.all(storage_pos_big_topo == cls.storage_pos_topo_vect):
            raise IncorrectPositionOfStorages(
                "Mismatch between storage_to_subid, "
                "storage_to_sub_pos and storage_pos_topo_vect"
            )

        # no empty bus: at least one element should be present on each bus
        if (cls.sub_info < 1).any():
            if not grid2op.Space.space_utils._WARNING_ISSUED_FOR_SUB_NO_ELEM:
                warnings.warn(
                    f"There are {np.sum(cls.sub_info < 1)} substations where  no 'controlable' elements "
                    f"are connected. These substations will be used in the computation of the powerflow "
                    f"(by the backend) but you will NOT be able to control anything on them."
                )
                grid2op.Space.space_utils._WARNING_ISSUED_FOR_SUB_NO_ELEM = True

        # redispatching / unit commitment
        if cls.redispatching_unit_commitment_available:
            cls._check_validity_dispatching_data()

        if cls.flexible_load_available:
            cls._check_validity_flexibile_loads()

        # shunt data
        if cls.shunts_data_available:
            cls._check_validity_shunt_data()

        # storage data
        cls._check_validity_storage_data()

        # alarm data
        cls._check_validity_alarm_data()

        # alert data
        cls._check_validity_alert_data()

    @classmethod
    def _check_validity_alarm_data(cls):
        if cls.dim_alarms == 0:
            # no alarm data
            assert (
                cls.alarms_area_names == []
            ), "No alarm data is provided, yet cls.alarms_area_names != []"
            assert (
                cls.alarms_lines_area == {}
            ), "No alarm data is provided, yet cls.alarms_lines_area != {}"
            assert (
                cls.alarms_area_lines == []
            ), "No alarm data is provided, yet cls.alarms_area_lines != []"
        elif cls.dim_alarms < 0:
            raise EnvError(
                f"The number of areas for the alarm feature should be >= 0. It currently is {cls.dim_alarms}"
            )
        else:
            assert cls.assistant_warning_type == "zonal"
            
            # the "alarm" feature is supported
            assert isinstance(
                cls.alarms_area_names, (list, tuple)
            ), "cls.alarms_area_names should be a list or a tuple"
            assert isinstance(
                cls.alarms_lines_area, dict
            ), "cls.alarms_lines_area should be a dict"
            assert isinstance(
                cls.alarms_area_lines, (list, tuple)
            ), "cls.alarms_area_lines should be a list or a tuple"
            assert (
                len(cls.alarms_area_names) == cls.dim_alarms
            ), "len(cls.alarms_area_names) != cls.dim_alarms"
            names_to_id = {nm: id_ for id_, nm in enumerate(cls.alarms_area_names)}

            # check that information in alarms_lines_area and alarms_area_lines match
            for l_nm, li_area in cls.alarms_lines_area.items():
                for area_nm in li_area:
                    area_id = names_to_id[area_nm]
                    all_lines_this_area = cls.alarms_area_lines[area_id]
                    assert l_nm in all_lines_this_area, (
                        f'line "{l_nm}" is said to belong to area "{area_nm}" '
                        f"in cls.alarms_lines_area yet when looking for the lines in "
                        f"this "
                        f"area in cls.alarms_area_lines, this line is not in there"
                    )

            for area_id, all_lines_this_area in enumerate(cls.alarms_area_lines):
                area_nm = cls.alarms_area_names[area_id]
                for l_nm in all_lines_this_area:
                    assert area_nm in cls.alarms_lines_area[l_nm], (
                        f'line "{l_nm}" is said to belong to area '
                        f'"{area_nm}" '
                        f"in cls.alarms_area_lines yet when looking for "
                        f"the areas where this line belong in "
                        f"cls.alarms_lines_area it appears it does not "
                        f"belong there."
                    )

            # now check that all lines are in at least one area
            for line, li_area in cls.alarms_lines_area.items():
                # check that all lines in the grid are in at least one area
                if not li_area:
                    raise EnvError(
                        f"Line (on the grid) named {line} is not in any area. This is not supported at "
                        f"the moment"
                    )
            # finally check that all powerlines are represented in the dictionary:
            for l_nm in cls.name_line:
                if l_nm not in cls.alarms_lines_area:
                    raise EnvError(
                        f'The powerline "{l_nm}" is not in cls.alarms_lines_area'
                    )

    @classmethod
    def _check_validity_alert_data(cls):
        # TODO remove assert and raise Grid2opExcpetion instead
        if cls.dim_alerts == 0:
            # no alert data
            assert (
                cls.alertable_line_names == []
            ), "No alert data is provided, yet cls.alertable_line_names != []"
            assert (
               len(cls.alertable_line_ids) == 0
            ), "No alert data is provided, yet len(cls.alertable_line_ids) != 0"
        elif cls.dim_alerts < 0:
            raise EnvError(
                f"The number of lines for the alert feature should be >= 0. It currently is {cls.dim_alerts}"
            )
        else:
            assert cls.assistant_warning_type == "by_line"
            # the "alert" feature is supported
            assert isinstance(
                cls.alertable_line_names, list
            ), "cls.alertable_line_names should be a list"
            assert (
                len(cls.alertable_line_names) == cls.dim_alerts
            ), "len(cls.alertable_line_names) != cls.dim_alerts"
            
            try:
                cls.alertable_line_ids = np.array(cls.alertable_line_ids).astype(dt_int)
            except Exception as exc_:
                raise EnvError(f"Impossible to convert alertable_line_ids "
                               f"to an array of int with error {exc_}")
            
    @classmethod
    def _check_validity_storage_data(cls):
        if cls.storage_type is None:
            raise IncorrectNumberOfStorages("self.storage_type is None")
        if cls.storage_Emax is None:
            raise IncorrectNumberOfStorages("self.storage_Emax is None")
        if cls.storage_Emin is None:
            raise IncorrectNumberOfStorages("self.storage_Emin is None")
        if cls.storage_max_p_prod is None:
            raise IncorrectNumberOfStorages("self.storage_max_p_prod is None")
        if cls.storage_max_p_absorb is None:
            raise IncorrectNumberOfStorages("self.storage_max_p_absorb is None")
        if cls.storage_marginal_cost is None:
            raise IncorrectNumberOfStorages("self.storage_marginal_cost is None")
        if cls.storage_loss is None:
            raise IncorrectNumberOfStorages("self.storage_loss is None")
        if cls.storage_discharging_efficiency is None:
            raise IncorrectNumberOfStorages(
                "self.storage_discharging_efficiency is None"
            )
        if cls.storage_charging_efficiency is None:
            raise IncorrectNumberOfStorages("self.storage_charging_efficiency is None")

        if cls.n_storage == 0:
            # no more check to perform is there is no storage
            return

        if cls.storage_type.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_type.shape[0] != self.n_storage"
            )
        if cls.storage_Emax.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_Emax.shape[0] != self.n_storage"
            )
        if cls.storage_Emin.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_Emin.shape[0] != self.n_storage"
            )
        if cls.storage_max_p_prod.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_max_p_prod.shape[0] != self.n_storage"
            )
        if cls.storage_max_p_absorb.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_max_p_absorb.shape[0] != self.n_storage"
            )
        if cls.storage_marginal_cost.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_marginal_cost.shape[0] != self.n_storage"
            )
        if cls.storage_loss.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_loss.shape[0] != self.n_storage"
            )
        if cls.storage_discharging_efficiency.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_discharging_efficiency.shape[0] != self.n_storage"
            )
        if cls.storage_charging_efficiency.shape[0] != cls.n_storage:
            raise IncorrectNumberOfStorages(
                "self.storage_charging_efficiency.shape[0] != self.n_storage"
            )

        if (~np.isfinite(cls.storage_Emax)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_Emax))")
        if (~np.isfinite(cls.storage_Emin)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_Emin))")
        if (~np.isfinite(cls.storage_max_p_prod)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_max_p_prod))")
        if (~np.isfinite(cls.storage_max_p_absorb)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_max_p_absorb))")
        if (~np.isfinite(cls.storage_marginal_cost)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_marginal_cost))")
        if (~np.isfinite(cls.storage_loss)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_loss))")
        if (~np.isfinite(cls.storage_charging_efficiency)).any():
            raise BackendError("np.any(~np.isfinite(self.storage_charging_efficiency))")
        if (~np.isfinite(cls.storage_discharging_efficiency)).any():
            raise BackendError(
                "np.any(~np.isfinite(self.storage_discharging_efficiency))"
            )

        if (cls.storage_Emax < cls.storage_Emin).any():
            tmp = (cls.storage_Emax < cls.storage_Emin).nonzero()[0]
            raise BackendError(
                f"storage_Emax < storage_Emin for storage units with ids: {tmp}"
            )
        if (cls.storage_Emax < 0.0).any():
            tmp = (cls.storage_Emax < 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_Emax < 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_Emin < 0.0).any():
            tmp = (cls.storage_Emin < 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_Emin < 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_max_p_prod < 0.0).any():
            tmp = (cls.storage_max_p_prod < 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_max_p_prod < 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_max_p_absorb < 0.0).any():
            tmp = (cls.storage_max_p_absorb < 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_max_p_absorb < 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_loss < 0.0).any():
            tmp = (cls.storage_loss < 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_loss < 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_discharging_efficiency <= 0.0).any():
            tmp = (cls.storage_discharging_efficiency <= 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_discharging_efficiency <= 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_discharging_efficiency > 1.0).any():
            tmp = (cls.storage_discharging_efficiency > 1.0).nonzero()[0]
            raise BackendError(
                f"self.storage_discharging_efficiency > 1. for storage units with ids: {tmp}"
            )
        if (cls.storage_charging_efficiency < 0.0).any():
            tmp = (cls.storage_charging_efficiency < 0.0).nonzero()[0]
            raise BackendError(
                f"self.storage_charging_efficiency < 0. for storage units with ids: {tmp}"
            )
        if (cls.storage_charging_efficiency > 1.0).any():
            tmp = (cls.storage_charging_efficiency > 1.0).nonzero()[0]
            raise BackendError(
                f"self.storage_charging_efficiency > 1. for storage units with ids: {tmp}"
            )
        if (cls.storage_loss > cls.storage_max_p_absorb).any():
            tmp = (cls.storage_loss > cls.storage_max_p_absorb).nonzero()[0]
            raise BackendError(
                f"Some storage units are such that their loss (self.storage_loss) is higher "
                f"than the maximum power at which they can be charged (self.storage_max_p_absorb). "
                f"Such storage units are doomed to discharged (due to losses) without anything "
                f"being able to charge them back. This really un interesting behaviour is not "
                f"supported by grid2op. Please check storage data for units {tmp}"
            )

    @classmethod
    def _check_validity_shunt_data(cls):
        if cls.n_shunt is None:
            raise IncorrectNumberOfElements(
                'Backend is supposed to support shunts, but "n_shunt" is not set.'
            )
        if cls.name_shunt is None:
            raise IncorrectNumberOfElements(
                'Backend is supposed to support shunts, but "name_shunt" is not set.'
            )
        if cls.shunt_to_subid is None:
            raise IncorrectNumberOfElements(
                'Backend is supposed to support shunts, but "shunt_to_subid" is not set.'
            )

        if not isinstance(cls.name_shunt, np.ndarray):
            try:
                cls.name_shunt = np.array(cls.name_shunt)
                cls.name_shunt = cls.name_shunt.astype(np.str_)
            except Exception as exc:
                raise EnvError(
                    'name_shunt should be convertible to a numpy array with dtype "str".'
                )

        if not isinstance(cls.shunt_to_subid, np.ndarray):
            try:
                cls.shunt_to_subid = np.array(cls.shunt_to_subid)
                cls.shunt_to_subid = cls.shunt_to_subid.astype(dt_int)
            except Exception as e:
                raise EnvError(
                    'shunt_to_subid should be convertible to a numpy array with dtype "int".'
                )

        if cls.name_shunt.shape[0] != cls.n_shunt:
            raise IncorrectNumberOfElements(
                'Backend is supposed to support shunts, but "name_shunt" has not '
                '"n_shunt" elements.'
            )
        if cls.shunt_to_subid.shape[0] != cls.n_shunt:
            raise IncorrectNumberOfElements(
                'Backend is supposed to support shunts, but "shunt_to_subid" has not '
                '"n_shunt" elements.'
            )
        if cls.n_shunt > 0:
            # check the substation id only if there are shunt
            if np.min(cls.shunt_to_subid) < 0:
                raise EnvError("Some shunt is connected to a negative substation id.")
            if np.max(cls.shunt_to_subid) > cls.n_sub:
                raise EnvError(
                    "Some shunt is supposed to be connected to substations with id {} which"
                    "is greater than the number of substations of the grid, which is {}."
                    "".format(np.max(cls.shunt_to_subid), cls.n_sub)
                )

    @classmethod
    def _check_validity_dispatching_data(cls):
        if cls.gen_type is None:
            raise InvalidRedispatching(
                "Impossible to recognize the type of generators (gen_type) when "
                "redispatching is supposed to be available."
            )
        if cls.gen_pmin is None:
            raise InvalidRedispatching(
                "Impossible to recognize the pmin of generators (gen_pmin) when "
                "redispatching is supposed to be available."
            )
        if cls.gen_pmax is None:
            raise InvalidRedispatching(
                "Impossible to recognize the pmax of generators (gen_pmax) when "
                "redispatching is supposed to be available."
            )
        if cls.gen_redispatchable is None:
            raise InvalidRedispatching(
                "Impossible to know which generator can be dispatched (gen_redispatchable)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_max_ramp_up is None:
            raise InvalidRedispatching(
                "Impossible to recognize the ramp up of generators (gen_max_ramp_up)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_max_ramp_down is None:
            raise InvalidRedispatching(
                "Impossible to recognize the ramp up of generators (gen_max_ramp_down)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_min_uptime is None:
            raise InvalidRedispatching(
                "Impossible to recognize the min uptime of generators (gen_min_uptime)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_min_downtime is None:
            raise InvalidRedispatching(
                "Impossible to recognize the min downtime of generators (gen_min_downtime)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_cost_per_MW is None:
            raise InvalidRedispatching(
                "Impossible to recognize the marginal costs of generators (gen_cost_per_MW)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_startup_cost is None:
            raise InvalidRedispatching(
                "Impossible to recognize the start up cost of generators (gen_startup_cost)"
                " when redispatching is supposed to be available."
            )
        if cls.gen_shutdown_cost is None:
            raise InvalidRedispatching(
                "Impossible to recognize the shut down cost of generators "
                "(gen_shutdown_cost) when redispatching is supposed to be available."
            )
        if cls.gen_renewable is None:
            raise InvalidRedispatching(
                "Impossible to recognize the whether generators comes from renewable energy "
                "sources "
                "(gen_renewable) when redispatching is supposed to be available."
            )

        if len(cls.gen_type) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the type of generators (gen_type) when "
                "redispatching is supposed to be available."
            )
        if len(cls.gen_pmin) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the pmin of generators (gen_pmin) when "
                "redispatching is supposed to be available."
            )
        if len(cls.gen_pmax) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the pmax of generators (gen_pmax) when "
                "redispatching is supposed to be available."
            )
        if len(cls.gen_redispatchable) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for which generator can be dispatched (gen_redispatchable)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_max_ramp_up) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the ramp up of generators (gen_max_ramp_up)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_max_ramp_down) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the ramp up of generators (gen_max_ramp_down)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_min_uptime) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the min uptime of generators (gen_min_uptime)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_min_downtime) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the min downtime of generators (gen_min_downtime)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_cost_per_MW) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the marginal costs of generators (gen_cost_per_MW)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_startup_cost) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the start up cost of generators (gen_startup_cost)"
                " when redispatching is supposed to be available."
            )
        if len(cls.gen_shutdown_cost) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the shut down cost of generators "
                "(gen_shutdown_cost) when redispatching is supposed to be available."
            )
        if len(cls.gen_renewable) != cls.n_gen:
            raise InvalidRedispatching(
                "Invalid length for the renewable flag vector"
                "(gen_renewable) when redispatching is supposed to be available."
            )

        if (cls.gen_min_uptime < 0).any():
            raise InvalidRedispatching(
                "Minimum uptime of generator (gen_min_uptime) cannot be negative"
            )
        if (cls.gen_min_downtime < 0).any():
            raise InvalidRedispatching(
                "Minimum downtime of generator (gen_min_downtime) cannot be negative"
            )

        for el in cls.gen_type:
            if not el in ["solar", "wind", "hydro", "thermal", "nuclear"]:
                raise InvalidRedispatching("Unknown generator type : {}".format(el))

        if (cls.gen_pmin < 0.0).any():
            raise InvalidRedispatching("One of the Pmin (gen_pmin) is negative")
        if (cls.gen_pmax < 0.0).any():
            raise InvalidRedispatching("One of the Pmax (gen_pmax) is negative")
        if (cls.gen_max_ramp_down < 0.0).any():
            raise InvalidRedispatching(
                "One of the ramp up (gen_max_ramp_down) is negative"
            )
        if (cls.gen_max_ramp_up < 0.0).any():
            raise InvalidRedispatching(
                "One of the ramp down (gen_max_ramp_up) is negative"
            )
        if (cls.gen_startup_cost < 0.0).any():
            raise InvalidRedispatching(
                "One of the start up cost (gen_startup_cost) is negative"
            )
        if (cls.gen_shutdown_cost < 0.0).any():
            raise InvalidRedispatching(
                "One of the start up cost (gen_shutdown_cost) is negative"
            )

        for el, type_ in zip(
            [
                "gen_type",
                "gen_pmin",
                "gen_pmax",
                "gen_redispatchable",
                "gen_max_ramp_up",
                "gen_max_ramp_down",
                "gen_min_uptime",
                "gen_min_downtime",
                "gen_cost_per_MW",
                "gen_startup_cost",
                "gen_shutdown_cost",
                "gen_renewable",
            ],
            [
                str,
                dt_float,
                dt_float,
                dt_bool,
                dt_float,
                dt_float,
                dt_int,
                dt_int,
                dt_float,
                dt_float,
                dt_float,
                dt_bool,
            ],
        ):
            if not isinstance(getattr(cls, el), np.ndarray):
                try:
                    setattr(cls, el, getattr(cls, el).astype(type_))
                except Exception as exc_:
                    raise InvalidRedispatching(
                        '{} should be convertible to a numpy array with error:\n "{}"'
                        "".format(el, exc_)
                    )
            if not np.issubdtype(getattr(cls, el).dtype, np.dtype(type_).type):
                try:
                    setattr(cls, el, getattr(cls, el).astype(type_))
                except Exception as exc_:
                    raise InvalidRedispatching(
                        "{} should be convertible data should be convertible to "
                        '{} with error: \n"{}"'.format(el, type_, exc_)
                    )
        if (
            cls.gen_max_ramp_up[cls.gen_redispatchable]
            > cls.gen_pmax[cls.gen_redispatchable]
        ).any():
            raise InvalidRedispatching(
                "Invalid maximum ramp for some generator (above pmax)"
            )

    @classmethod
    def _check_validity_flexibile_loads(cls):
        
        for attr_name in cls._li_attr_flex_load:
            attr = getattr(cls, attr_name, None)
            if attr is None:
                raise InvalidFlexibility(
                    f"Impossible to recognize the ({attr_name}) of loads when "
                    "flexibility is supposed to be available."
                )
            elif len(attr) != cls.n_load:
                raise InvalidFlexibility(
                    f"Invalid length for the ({attr_name}) of loads when "
                    "flexibility is supposed to be available."
                 )
            elif attr.dtype is not np.dtype(bool) and  (attr < 0).any():
                raise InvalidFlexibility(
                    f"One of the ({attr_name}) is negative"
                )
        
        for el, prim_type in zip(cls._li_attr_flex_load, cls._type_attr_flex_load):
            type_ = {float:dt_float, bool:dt_bool, int:dt_int, str:str}[prim_type]
            if not isinstance(getattr(cls, el), np.ndarray):
                try:
                    setattr(cls, el, getattr(cls, el).astype(type_))
                except Exception as exc_:
                    raise InvalidFlexibility(
                        '{} should be convertible to a numpy array with error:\n "{}"'
                        "".format(el, exc_)
                    )
            if not np.issubdtype(getattr(cls, el).dtype, np.dtype(type_).type):
                try:
                    setattr(cls, el, getattr(cls, el).astype(type_))
                except Exception as exc_:
                    raise InvalidFlexibility(
                        "{} should be convertible data should be convertible to "
                        '{} with error: \n"{}"'.format(el, type_, exc_)
                    )
        if (cls.load_max_ramp_up[cls.load_flexible] > cls.load_size[cls.load_flexible]).any():
            raise InvalidFlexibility("Invalid maximum ramp for some loads (above size of load)")
    
    @classmethod
    def attach_layout(cls, grid_layout):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            We do not recommend to "attach layout" outside of the environment. Please refer to the function
            :func:`grid2op.Environment.BaseEnv.attach_layout` for more information.

        grid layout is a dictionary with the keys the name of the substations, and the value the tuple of coordinates
        of each substations. No check are made it to ensure it is correct.

        Parameters
        ----------
        grid_layout: ``dict``
            See definition of :attr:`GridObjects.grid_layout` for more information.

        """
        cls.grid_layout = grid_layout

    @classmethod
    def set_env_name(cls, name):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Do not attempt in any case to modify the name of the environment once it has been loaded. If you
            do that, you might experience undefined behaviours, notably with the multi processing but not only.

        """
        cls.env_name = name

    @classmethod
    def _aux_init_grid_from_cls(cls, gridobj, name_res):
        import importlib
        # NB: these imports needs to be consistent with what is done in
        # base_env.generate_classes()
        super_module_nm, module_nm = os.path.split(gridobj._PATH_GRID_CLASSES)
        if module_nm == "_grid2op_classes":
            # legacy "experimental_read_from_local_dir"
            # issue was the module "_grid2op_classes" had the same name
            # regardless of the environment, so grid2op was "confused"
            env_path, env_nm = os.path.split(super_module_nm)
            if env_path not in sys.path:
                sys.path.append(env_path)
            super_supermodule = importlib.import_module(env_nm)
            module_nm = f"{env_nm}.{module_nm}"
            super_module_nm = super_supermodule
        
        if f"{module_nm}.{name_res}_file" in sys.modules:
            cls_res = getattr(sys.modules[f"{module_nm}.{name_res}_file"], name_res)
            # do not forget to create the cls_dict once and for all
            if cls_res._CLS_DICT is None:
                tmp = {}
                cls_res._make_cls_dict_extended(cls_res, tmp, as_list=False)
            return cls_res
        
        super_module = importlib.import_module(module_nm, super_module_nm)  # env/path/_grid2op_classes/
        module_all_classes = importlib.import_module(f"{module_nm}")  # module specific to the tmpdir created
        try:
            module = importlib.import_module(f".{name_res}_file", package=module_nm)  # module containing the definition of the class
        except ModuleNotFoundError:
            # in case we need to build the cache again if the module is not found the first time
            importlib.invalidate_caches()
            importlib.reload(super_module)
            module = importlib.import_module(f".{name_res}_file", package=module_nm)
        cls_res = getattr(module, name_res)
        # do not forget to create the cls_dict once and for all
        if cls_res._CLS_DICT is None:
            tmp = {}
            cls_res._make_cls_dict_extended(cls_res, tmp, as_list=False)
        return cls_res
    
    @classmethod
    def init_grid(cls, gridobj, force=False, extra_name=None, force_module=None, _local_dir_cls=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is done at the creation of the environment. Use of this class outside of this particular
            use is really dangerous and will lead to undefined behaviours. **Do not use this function**.

        Initialize this :class:`GridObjects` subclass with a provided class.

        It does not perform any check on the validity of the `gridobj` parameters, but it guarantees that  if `gridobj`
        is a valid grid, then the initialization will lead to a valid grid too.

        Parameters
        ----------
        gridobj: :class:`GridObjects`
            The representation of the powergrid

        force: ``bool``
            force the initialization of the class. By default if a class with the same name exists in `globals()`
            it does not initialize it. Setting "force=True" will bypass this check and update it accordingly.

        """
        # nothing to do now that the value are class member
        name_res = "{}_{}".format(cls.__name__, gridobj.env_name)
        if gridobj.glop_version != grid2op.__version__:
            name_res += f"_{gridobj.glop_version}"
        
        if gridobj._PATH_GRID_CLASSES is not None:
            # the configuration equires to initialize the classes from the local environment path
            # this might be usefull when using pickle module or multiprocessing on Windows for example
            my_class = GridObjects._build_cls_from_import(name_res, gridobj._PATH_GRID_CLASSES)
            if my_class is not None:
                return my_class
        
        if not gridobj.shunts_data_available:
            # if you import env for backend
            # with shunt and without shunt, then
            # there might be issues
            name_res += "_noshunt"
        
        # TODO n_busbar_per_sub different num per substations: if it's a vector, use some kind of hash of it
        # for the name of the class !
        if gridobj.n_busbar_per_sub != DEFAULT_N_BUSBAR_PER_SUB:
            # to be able to load same environment with
            # different `n_busbar_per_sub`
            name_res += f"_{gridobj.n_busbar_per_sub}"
                
        if _local_dir_cls is not None and gridobj._PATH_GRID_CLASSES is not None:
            # new in grid2op 1.10.3:
            # if I end up here it's because (done in base_env.generate_classes()):
            # 1) the first initial env has already been created
            # 2) I need to init the class from the files (and not from whetever else)
            # So i do it. And if that is the case, the files are created on the hard drive
            # AND the module is added to the path
            
            # check that it matches (security / consistency check)
            if not os.path.samefile(_local_dir_cls.name ,  gridobj._PATH_GRID_CLASSES):
                # in windows the string comparison fails because of things like "/", "\" or "\\"
                # this is why we use "samefile"
                raise EnvError(f"Unable to create the class: mismatch between "
                               f"_local_dir_cls ({_local_dir_cls.name}) and " 
                               f" _PATH_GRID_CLASSES ({gridobj._PATH_GRID_CLASSES})")
            return cls._aux_init_grid_from_cls(gridobj, name_res)
        elif gridobj._PATH_GRID_CLASSES is not None:
            # If I end up it's because the environment is created with already initialized
            # classes.
            return cls._aux_init_grid_from_cls(gridobj, name_res)
        
        # legacy behaviour: build the class "on the fly"
        # of new (>= 1.10.3 for the intial creation of the environment)
        if name_res in globals():
            if not force and _local_dir_cls is None:
                # no need to recreate the class, it already exists
                return globals()[name_res]
            else:
                # i recreate the variable
                del globals()[name_res]
            
        cls_attr_as_dict = {}
        GridObjects._make_cls_dict_extended(gridobj, cls_attr_as_dict, as_list=False)
        res_cls = type(name_res, (cls,), cls_attr_as_dict)
        if hasattr(cls, "_INIT_GRID_CLS") and cls._INIT_GRID_CLS is not None:
            # original class is already from an initialized environment, i keep track of it
            res_cls._INIT_GRID_CLS = cls._INIT_GRID_CLS
        else:
            # i am the original class from grid2op
            res_cls._INIT_GRID_CLS = cls
        
        res_cls._IS_INIT = True
        
        res_cls._compute_pos_big_topo_cls()
        res_cls.process_shunt_satic_data()
        compat_mode = res_cls.process_grid2op_compat()
        res_cls._check_convert_to_np_array()  # convert everything to numpy array
        if force_module is not None:
            res_cls.__module__ = force_module  # hack because otherwise it says "abc" which is not the case
            # best would be to have a look at https://docs.python.org/3/library/types.html
        
        if not compat_mode:
            # I can reuse the "cls" dictionnary as they did not changed
            if cls._CLS_DICT is not None:
                res_cls._CLS_DICT = cls._CLS_DICT
            if cls._CLS_DICT_EXTENDED is not None:
                res_cls._CLS_DICT_EXTENDED = cls._CLS_DICT_EXTENDED
        else:
            # I need to rewrite the _CLS_DICT and _CLS_DICT_EXTENDED
            # as the class has been modified with a "compatibility version" mode
            tmp = {}
            res_cls._make_cls_dict_extended(res_cls, tmp, as_list=False)
            
        # store the type created here in the "globals" to prevent the initialization of the same class over and over
        globals()[name_res] = res_cls
        del res_cls
        return globals()[name_res]

    @classmethod
    def _get_grid2op_version_as_version_obj(cls):
        if cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            glop_ver = version.parse("0.0.0")
        else:
            glop_ver = version.parse(cls.glop_version)
        return glop_ver
        
    @classmethod
    def process_grid2op_compat(cls):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is done at the creation of the environment. Use of this class outside of this particular
            use is really dangerous and will lead to undefined behaviours. **Do not use this function**.
            
        This is called when the class is initialized, with `init_grid` to broadcast grid2op compatibility feature.
        
        This function can be overloaded, but in this case it's best to call this original method too.

        """
        res = False
        glop_ver = cls._get_grid2op_version_as_version_obj()
        
        if cls.glop_version == cls.BEFORE_COMPAT_VERSION:
            # oldest version: no storage and no curtailment available
            cls._aux_process_old_compat()
            res = True
            
        if glop_ver < version.parse("1.6.0"):
            # this feature did not exist before.
            cls.dim_alarms = 0
            cls.assistant_warning_type = None
            res = True
            
        if glop_ver < version.parse("1.9.1"):
            # this feature did not exists before
            cls.dim_alerts = 0 
            cls.alertable_line_names = []
            cls.alertable_line_ids = []
            res = True
            
        if glop_ver < version.parse("1.10.0.dev0"):
            # this feature did not exists before
            # I need to set it to the default if set elsewhere
            cls.n_busbar_per_sub = DEFAULT_N_BUSBAR_PER_SUB
            res = True

        if glop_ver < version.parse("1.10.4.dev0"):
            # Flexibility did not exist before
            # Affects shape of vector representation 
            cls.flexible_load_available = False
            cls._aux_process_pre_flexibility()
            res = True
            
        if res:
            cls._reset_cls_dict()  # forget the previous class (stored as dict)
        return res

    @classmethod
    def _aux_fix_topo_vect_removed_storage(cls):
        if cls.n_storage == 0:
            return
        
        stor_locs = [pos for pos in cls.storage_pos_topo_vect]
        for stor_loc in sorted(stor_locs, reverse=True):
            for vect in [
                cls.load_pos_topo_vect,
                cls.gen_pos_topo_vect,
                cls.line_or_pos_topo_vect,
                cls.line_ex_pos_topo_vect,
            ]:
                vect[vect >= stor_loc] -= 1

        # deals with the "sub_pos" vector
        for sub_id in range(cls.n_sub):
            if (cls.storage_to_subid == sub_id).any():
                stor_ids = (cls.storage_to_subid == sub_id).nonzero()[0]
                stor_locs = cls.storage_to_sub_pos[stor_ids]
                for stor_loc in sorted(stor_locs, reverse=True):
                    for vect, sub_id_me in zip(
                        [
                            cls.load_to_sub_pos,
                            cls.gen_to_sub_pos,
                            cls.line_or_to_sub_pos,
                            cls.line_ex_to_sub_pos,
                        ],
                        [
                            cls.load_to_subid,
                            cls.gen_to_subid,
                            cls.line_or_to_subid,
                            cls.line_ex_to_subid,
                        ],
                    ):
                        vect[(vect >= stor_loc) & (sub_id_me == sub_id)] -= 1

        # remove storage from the number of element in the substation
        for sub_id in range(cls.n_sub):
            cls.sub_info[sub_id] -= (cls.storage_to_subid == sub_id).sum()
        # remove storage from the total number of element
        cls.dim_topo -= cls.n_storage

        # recompute this private member
        cls._topo_vect_to_sub = np.repeat(
            np.arange(cls.n_sub), repeats=cls.sub_info
        )

        new_grid_objects_types = cls.grid_objects_types
        new_grid_objects_types = new_grid_objects_types[
            new_grid_objects_types[:, cls.STORAGE_COL] == -1, :
        ]
        cls.grid_objects_types = 1 * new_grid_objects_types
        
    @classmethod
    def _aux_process_old_compat(cls):
        # remove "storage dependant attributes (topo_vect etc.) that are modified !"
        cls._aux_fix_topo_vect_removed_storage()
        # deactivate storage
        cls.set_no_storage()

    @classmethod
    def _aux_process_pre_flexibility(cls):
        # Remove flexibility
        flex_attrs = ["load_size", "load_flexible", 
                      "load_max_ramp_up", "load_max_ramp_down",
                      "load_min_uptime", "load_min_downtime",
                      "load_cost_per_MW"]
        for flex_attr in flex_attrs:
            if hasattr(cls, flex_attr):
                setattr(cls, flex_attr, np.zeros([], dtype=dt_int))
        if cls.attr_list_vect is not None:
            if "actual_flex" in cls.attr_list_vect:
                cls.attr_list_vect.remove("actual_flex")
            if "target_flex" in cls.attr_list_vect:
                cls.attr_list_vect.remove("target_flex")
            cls.attr_list_set = set(cls.attr_list_vect)
            
    @classmethod
    def get_obj_connect_to(cls, _sentinel=None, substation_id=None):
        """
        Get all the object connected to a given substation. This is particularly usefull if you want to know the
        names of the generator / load connected to a given substation, or which extremity etc.

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        substation_id: ``int``
            ID of the substation we want to inspect

        Returns
        -------
        res: ``dict``
            A dictionary with keys:

              - "loads_id": a vector giving the id of the loads connected to this substation, empty if none
              - "generators_id": a vector giving the id of the generators connected to this substation, empty if none
              - "lines_or_id": a vector giving the id of the origin side of the powerlines connected to this substation,
                empty if none
              - "lines_ex_id": a vector giving the id of the extermity side of the powerlines connected to this
                substation, empty if none.
              - "storages_id": a vector giving the id of the storage units connected at this substation.
              - "nb_elements" : number of elements connected to this substation

        Examples
        --------

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # get the vector representation of an observation:
            sub_id = 1
            dict_ = env.get_obj_connect_to(substation_id=sub_id)
            print("There are {} elements connected to this substation (not counting shunt)".format(
                  dict_["nb_elements"]))
            print("The names of the loads connected to substation {} are: {}".format(
                   sub_id, env.name_load[dict_["loads_id"]]))
            print("The names of the generators connected to substation {} are: {}".format(
                   sub_id, env.name_gen[dict_["generators_id"]]))
            print("The powerline whose origin side is connected to substation {} are: {}".format(
                   sub_id, env.name_line[dict_["lines_or_id"]]))
            print("The powerline whose extremity side is connected to substation {} are: {}".format(
                   sub_id, env.name_line[dict_["lines_ex_id"]]))
            print("The storage units connected to substation {} are: {}".format(
                   sub_id, env.name_line[dict_["storages_id"]]))

        """
        if _sentinel is not None:
            raise Grid2OpException(
                "get_obj_connect_to should be used only with key-word arguments"
            )

        if substation_id is None:
            raise Grid2OpException(
                "You ask the composition of a substation without specifying its id."
                'Please provide "substation_id"'
            )
        if substation_id >= len(cls.sub_info):
            raise Grid2OpException(
                'There are no substation of id "substation_id={}" in this grid.'
                "".format(substation_id)
            )
        res = {
            "loads_id": (cls.load_to_subid == substation_id).nonzero()[0],
            "generators_id": (cls.gen_to_subid == substation_id).nonzero()[0],
            "lines_or_id": (cls.line_or_to_subid == substation_id).nonzero()[0],
            "lines_ex_id": (cls.line_ex_to_subid == substation_id).nonzero()[0],
            "storages_id": (cls.storage_to_subid == substation_id).nonzero()[0],
            "nb_elements": cls.sub_info[substation_id],
        }
        return res

    @classmethod
    def get_powerline_id(cls, sub_id: int) -> np.ndarray:
        """
        Return the id of all powerlines connected to the substation `sub_id`
        either "or" side or "ex" side
        
        Parameters
        -----------
        sub_id: `int`
            The id of the substation concerned
            
        Returns
        -------     
        res: np.ndarray, int
            The id of all powerlines connected to this substation (either or side or ex side)
        
        Examples
        --------

        To get the id of all powerlines connected to substation with id 1, 
        you can do:
        
        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")
            
            all_lines_conn_to_sub_id_1 = type(env).get_powerline_id(1)   
                
        """
        powerlines_or_id = cls.line_or_to_sub_pos[
            cls.line_or_to_subid == sub_id
        ]
        powerlines_ex_id = cls.line_ex_to_sub_pos[
            cls.line_ex_to_subid == sub_id
        ]
        powerlines_id = np.concatenate((powerlines_or_id, powerlines_ex_id))
        return powerlines_id
    
    @classmethod
    def get_obj_substations(cls, _sentinel=None, substation_id=None):
        """
        Return the object connected as a substation in form of a numpy array instead of a dictionary (as
        opposed to :func:`GridObjects.get_obj_connect_to`).

        This format is particularly useful for example if you want to know the number of generator connected
        to a given substation for example (see section examples).

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        substation_id: ``int``
            ID of the substation we want to inspect

        Returns
        -------
        res: ``numpy.ndarray``
            A matrix with as many rows as the number of element of the substation and 6 columns:

              1. column 0: the id of the substation
              2. column 1: -1 if this object is not a load, or `LOAD_ID` if this object is a load (see example)
              3. column 2: -1 if this object is not a generator, or `GEN_ID` if this object is a generator (see example)
              4. column 3: -1 if this object is not the origin side of a line, or `LOR_ID` if this object is the
                 origin side of a powerline(see example)
              5. column 4: -1 if this object is not a extremity side, or `LEX_ID` if this object is the extremity
                 side of a powerline
              6. column 5: -1 if this object is not a storage unit, or `STO_ID` if this object is one

        Examples
        --------

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # get the vector representation of an observation:
            sub_id = 1
            mat = env.get_obj_substations(substation_id=sub_id)

            # the first element of the substation is:
            mat[0,:]
            # array([ 1, -1, -1, -1,  0, -1], dtype=int32)
            # we know it's connected to substation 1... no kidding...
            # we can also get that:
            # 1. this is not a load (-1 at position 1 - so 2nd component)
            # 2. this is not a generator (-1 at position 2 - so 3rd component)
            # 3. this is not the origin side of a powerline (-1 at position 3)
            # 4. this is the extremity side of powerline 0 (there is a 0 at position 4)
            # 5. this is not a storage unit (-1 at position 5 - so last component)

            # likewise, the second element connected at this substation is:
            mat[1,:]
            # array([ 1, -1, -1,  2, -1, -1], dtype=int32)
            # it represents the origin side of powerline 2

            # the 5th element connected at this substation is:
            mat[4,:]
            # which is equal to  array([ 1, -1,  0, -1, -1, -1], dtype=int32)
            # so it's represents a generator, and this generator has the id 0

            # the 6th element connected at this substation is:
            mat[5,:]
            # which is equal to  array([ 1, 0,  -1, -1, -1, -1], dtype=int32)
            # so it's represents a generator, and this generator has the id 0

            # and, last example, if you want to count the number of generator connected at this
            # substation you can
            is_gen = mat[:,env.GEN_COL] != -1  # a boolean vector saying ``True`` if the object is a generator
            nb_gen_this_substation = np.sum(is_gen)

        """
        if _sentinel is not None:
            raise Grid2OpException(
                "get_obj_substations should be used only with key-word arguments"
            )

        if substation_id is None:
            raise Grid2OpException(
                "You ask the composition of a substation without specifying its id."
                'Please provide "substation_id"'
            )
        if substation_id >= len(cls.sub_info):
            raise Grid2OpException(
                'There are no substation of id "substation_id={}" in this grid.'
                "".format(substation_id)
            )

        dict_ = cls.get_obj_connect_to(substation_id=substation_id)
        res = np.full((dict_["nb_elements"], 6), fill_value=-1, dtype=dt_int)
        # 0 -> load, 1-> gen, 2 -> lines_or, 3 -> lines_ex
        res[:, cls.SUB_COL] = substation_id
        res[cls.load_to_sub_pos[dict_["loads_id"]], cls.LOA_COL] = dict_["loads_id"]
        res[cls.gen_to_sub_pos[dict_["generators_id"]], cls.GEN_COL] = dict_[
            "generators_id"
        ]
        res[cls.line_or_to_sub_pos[dict_["lines_or_id"]], cls.LOR_COL] = dict_[
            "lines_or_id"
        ]
        res[cls.line_ex_to_sub_pos[dict_["lines_ex_id"]], cls.LEX_COL] = dict_[
            "lines_ex_id"
        ]
        res[cls.storage_to_sub_pos[dict_["storages_id"]], cls.STORAGE_COL] = dict_[
            "storages_id"
        ]
        return res

    @classmethod
    def get_lines_id(cls, _sentinel=None, from_=None, to_=None):
        """
        Returns the list of all the powerlines id in the backend going from `from_` to `to_`

        Parameters
        ----------
        _sentinel: ``None``
            Internal, do not use

        from_: ``int``
            Id the substation to which the origin side of the powerline to look for should be connected to

        to_: ``int``
            Id the substation to which the extremity side of the powerline to look for should be connected to

        Returns
        -------
        res: ``list``
            Id of the powerline looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match is found.


        Examples
        --------
        It can be used like:

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            l_ids = env.get_lines_id(from_=0, to_=1)
            print("The powerlines connecting substation 0 to substation 1 have for ids: {}".format(l_ids))

        """
        res = []
        if from_ is None:
            raise BackendError(
                "ObservationSpace.get_lines_id: impossible to look for a powerline with no origin "
                'substation. Please modify "from_" parameter'
            )
        if to_ is None:
            raise BackendError(
                "ObservationSpace.get_lines_id: impossible to look for a powerline with no extremity "
                'substation. Please modify "to_" parameter'
            )

        for i, (ori, ext) in enumerate(
            zip(cls.line_or_to_subid, cls.line_ex_to_subid)
        ):
            if ori == from_ and ext == to_:
                res.append(i)

        if not res:  # res is empty here
            raise BackendError(
                "ObservationSpace.get_line_id: impossible to find a powerline with connected at "
                "origin at {} and extremity at {}".format(from_, to_)
            )

        return res

    @classmethod
    def get_generators_id(cls, sub_id):
        """
        Returns the list of all generators id in the backend connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the generator

        Returns
        -------
        res: ``list``
            Id of the generators looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match is found.


        Examples
        --------
        It can be used like:

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            g_ids = env.get_generators_id(sub_id=1)
            print("The generators connected to substation 1 have for ids: {}".format(g_ids))

        """
        res = []
        if sub_id is None:
            raise BackendError(
                "GridObjects.get_generators_id: impossible to look for a generator not connected to any substation. "
                'Please modify "sub_id" parameter'
            )

        for i, s_id_gen in enumerate(cls.gen_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if not res:  # res is empty here
            raise BackendError(
                "GridObjects.get_generators_id: impossible to find a generator connected at "
                "substation {}".format(sub_id)
            )

        return res

    @classmethod
    def get_loads_id(cls, sub_id):
        """
        Returns the list of all loads id in the backend connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the generator

        Returns
        -------
        res: ``list``
            Id of the loads looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match found.

        Examples
        --------
        It can be used like:

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            c_ids = env.get_loads_id(sub_id=1)
            print("The loads connected to substation 1 have for ids: {}".format(c_ids))

        """
        res = []
        if sub_id is None:
            raise BackendError(
                "GridObjects.get_loads_id: impossible to look for a load not connected to any substation. "
                'Please modify "sub_id" parameter'
            )

        for i, s_id_gen in enumerate(cls.load_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if not res:  # res is empty here
            raise BackendError(
                "GridObjects.get_loads_id: impossible to find a load connected at substation {}".format(
                    sub_id
                )
            )

        return res

    @classmethod
    def get_storages_id(cls, sub_id):
        """
        Returns the list of all storages element (battery or damp) id in the grid connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the storage unit

        Returns
        -------
        res: ``list``
            Id of the storage elements looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match found.

        Examples
        --------
        It can be used like:

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            sto_ids = env.get_storages_id(sub_id=1)
            print("The loads connected to substation 1 have for ids: {}".format(c_ids))

        """
        res = []
        if sub_id is None:
            raise BackendError(
                "GridObjects.get_storages_id: impossible to look for a load not connected to any substation. "
                'Please modify "sub_id" parameter'
            )

        for i, s_id_gen in enumerate(cls.storage_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if not res:  # res is empty here
            raise BackendError(
                "GridObjects.get_storages_id: impossible to find a storage unit connected at substation {}".format(
                    sub_id
                )
            )

        return res

    @classmethod
    def topo_vect_element(cls, topo_vect_id: int) -> Dict[Literal["load_id", "gen_id", "line_id", "storage_id", "line_or_id", "line_ex_id", "sub_id"],
                                                          Union[int, Dict[Literal["or", "ex"], int]]]:
        """
        This function aims to be the "opposite" of the
        `cls.xxx_pos_topo_vect` (**eg** `cls.load_pos_topo_vect`)
        
        You give it an id in the topo_vect (*eg* 10) and it gives you 
        information about which element it is. More precisely, if 
        `type(env).topo_vect[topo_vect_id]` is:
        
        - a **load** then it will return `{'load_id': load_id}`, with `load_id`
          being such that `type(env).load_pos_topo_vect[load_id] == topo_vect_id`
        - a **generator** then it will return `{'gen_id': gen_id}`, with `gen_id`
          being such that `type(env).gen_pos_topo_vect[gen_id] == topo_vect_id`
        - a **storage** then it will return `{'storage_id': storage_id}`, with `storage_id`
          being such that `type(env).storage_pos_topo_vect[storage_id] == topo_vect_id`
        - a **line** (origin side) then it will return `{'line_id': {'or': line_id}, 'line_or_id': line_id}`, 
          with `line_id`
          being such that `type(env).line_or_pos_topo_vect[line_id] == topo_vect_id`
        - a **line** (ext side) then it will return `{'line_id': {'ex': line_id}, 'line_ex_id': line_id}`, 
          with `line_id`
          being such that `type(env).line_or_pos_topo_vect[line_id] == topo_vect_id`
          
        .. seealso::
            The attributes :attr:`GridObjects.load_pos_topo_vect`, :attr:`GridObjects.gen_pos_topo_vect`,
            :attr:`GridObjects.storage_pos_topo_vect`, :attr:`GridObjects.line_or_pos_topo_vect` and
            :attr:`GridObjects.line_ex_pos_topo_vect` to do the opposite.
            
            And you can also have a look at :attr:`GridObjects.grid_objects_types`
          
        Parameters
        ----------
        topo_vect_id: ``int``
            The element of the topo vect to which you want more information.

        Returns
        -------
        res: ``dict``
            See details in the description

        Examples
        --------
        It can be used like:

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            env_cls = type(env)  # or `type(act)` or` type(obs)` etc. or even `env.topo_vect_element(...)` or `obs.topo_vect_element(...)`
            for load_id, pos_topo_vect in enumerate(env_cls.load_pos_topo_vect):
                res = env_cls.topo_vect_element(pos_topo_vect)
                assert "load_id" in res
                assert res["load_id"] == load_id
                
            for gen_id, pos_topo_vect in enumerate(env_cls.gen_pos_topo_vect):
                res = env_cls.topo_vect_element(pos_topo_vect)
                assert "gen_id" in res
                assert res["gen_id"] == gen_id
                
            for sto_id, pos_topo_vect in enumerate(env_cls.storage_pos_topo_vect):
                res = env_cls.topo_vect_element(pos_topo_vect)
                assert "storage_id" in res
                assert res["storage_id"] == sto_id
                
            for line_id, pos_topo_vect in enumerate(env_cls.line_or_pos_topo_vect):
                res = env_cls.topo_vect_element(pos_topo_vect)
                assert "line_id" in res
                assert res["line_id"] == {"or": line_id}
                assert "line_or_id" in res
                assert res["line_or_id"] == line_id
                
            for line_id, pos_topo_vect in enumerate(env_cls.line_ex_pos_topo_vect):
                res = env_cls.topo_vect_element(pos_topo_vect)
                assert "line_id" in res
                assert res["line_id"] == {"ex": line_id}
                assert "line_ex_id" in res
                assert res["line_ex_id"] == line_id
                
        """
        elt = cls.grid_objects_types[topo_vect_id]
        res = {"sub_id": int(elt[cls.SUB_COL])}
        if elt[cls.LOA_COL] != -1:
            res["load_id"] = int(elt[cls.LOA_COL])
            return res
        if elt[cls.GEN_COL] != -1:
            res["gen_id"] = int(elt[cls.GEN_COL])
            return res
        if elt[cls.STORAGE_COL] != -1:
            res["storage_id"] = int(elt[cls.STORAGE_COL])
            return res
        if elt[cls.LOR_COL] != -1:
            res["line_or_id"] = int(elt[cls.LOR_COL])
            res["line_id"] = {"or": int(elt[cls.LOR_COL])}
            return res
        if elt[cls.LEX_COL] != -1:
            res["line_ex_id"] = int(elt[cls.LEX_COL])
            res["line_id"] = {"ex": int(elt[cls.LEX_COL])}
            return res
        raise Grid2OpException(f"Unknown element at position {topo_vect_id}")

    @staticmethod
    def _make_cls_dict(cls, res, as_list=True, copy_=True, _topo_vect_only=False):
        """ 
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        
        NB: `cls` can be here a class or an object of a class...
        
        Notes
        -------
        _topo_vect_only: this function is called once when the backend is initialized in `backend.load_grid`  
        (in `backend._compute_pos_big_topo`) and then once when everything is set up 
        (after redispatching, flexibility and storage data are loaded).
        
        This is why I need the `_topo_vect_only` flag that tells this function when it's called only for 
        `topo_vect` related attributed
        
        """
        if cls._CLS_DICT is not None and not as_list and not _topo_vect_only:
            # speed optimization: it has already been computed, so 
            # I reuse it (class attr are const)
            for k, v in cls._CLS_DICT.items():
                if copy_:
                    res[k] = copy.deepcopy(v)
                else:
                    res[k] = v
            return

        if not _topo_vect_only:
            # all the attributes bellow are not needed for the "first call"
            # to this function when the elements are put together in the topo_vect.
            # Indeed, at this stage (first call in the backend.load_grid) these
            # attributes are not (necessary) loaded yet
            save_to_dict(res, cls, "glop_version", str, copy_)
            res["_PATH_GRID_CLASSES"] = cls._PATH_GRID_CLASSES  # i do that manually for more control
            save_to_dict(res, cls, "n_busbar_per_sub", str, copy_)
        
        save_to_dict(
            res,
            cls,
            "name_gen",
            (lambda arr: [str(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "name_load",
            (lambda li: [str(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "name_line",
            (lambda li: [str(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "name_sub",
            (lambda li: [str(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "name_storage",
            (lambda li: [str(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(res, cls, "env_name", str, copy_)

        save_to_dict(
            res,
            cls,
            "sub_info",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )

        save_to_dict(
            res,
            cls,
            "load_to_subid",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "gen_to_subid",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "line_or_to_subid",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "line_ex_to_subid",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "storage_to_subid",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )

        save_to_dict(
            res,
            cls,
            "load_to_sub_pos",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "gen_to_sub_pos",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "line_or_to_sub_pos",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "line_ex_to_sub_pos",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "storage_to_sub_pos",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )

        save_to_dict(
            res,
            cls,
            "load_pos_topo_vect",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "gen_pos_topo_vect",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "line_or_pos_topo_vect",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "line_ex_pos_topo_vect",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            cls,
            "storage_pos_topo_vect",
            (lambda li: [int(el) for el in li]) if as_list else None,
            copy_,
        )

        # shunts (not in topo vect but still usefull)
        if cls.shunts_data_available:
            save_to_dict(
                res,
                cls,
                "name_shunt",
                (lambda li: [str(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "shunt_to_subid",
                (lambda li: [int(el) for el in li]) if as_list else None,
                copy_,
            )
        else:
            res["name_shunt"] = None
            res["shunt_to_subid"] = None
            
        if not _topo_vect_only:
            # all the attributes bellow are not needed for the "first call"
            # to this function when the elements are put together in the topo_vect.
            # Indeed, at this stage (first call in the backend.load_grid) these
            # attributes are not loaded yet
            
            # Redispatching
            if cls.redispatching_unit_commitment_available:
                for nm_attr, type_attr in zip(cls._li_attr_disp, cls._type_attr_disp):
                    save_to_dict(
                        res,
                        cls,
                        nm_attr,
                        (lambda li: [type_attr(el) for el in li]) if as_list else None,
                        copy_,
                    )
            else:
                for nm_attr in cls._li_attr_disp:
                    res[nm_attr] = None
            
            # Flexibility
            if cls._get_grid2op_version_as_version_obj() >= version.parse("1.10.4.dev0"):
                for nm_attr, type_attr in zip(cls._li_attr_flex_load, cls._type_attr_flex_load):
                    save_to_dict(
                        res,
                        cls,
                        nm_attr,
                        (lambda li: [type_attr(el) for el in li]) if as_list else None,
                        copy_,
                    )
            # else:
            #     for nm_attr in cls._li_attr_flex_load:
            #         res[nm_attr] = None

            # Layout (position of substation on a map of the grid)
            if cls.grid_layout is not None:
                save_to_dict(
                    res,
                    cls,
                    "grid_layout",
                    (lambda gl: {str(k): [float(x), float(y)] for k, (x, y) in gl.items()})
                    if as_list
                    else None,
                    copy_,
                )
            else:
                res["grid_layout"] = None

            # Storage Data
            save_to_dict(
                res,
                cls,
                "storage_type",
                (lambda li: [str(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_Emax",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_Emin",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_max_p_prod",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_max_p_absorb",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_marginal_cost",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_loss",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_charging_efficiency",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "storage_discharging_efficiency",
                (lambda li: [float(el) for el in li]) if as_list else None,
                copy_,
            )

            # Alert or Alarm
            if cls.assistant_warning_type is not None:
                res["assistant_warning_type"] = str(cls.assistant_warning_type)
            else:
                res["assistant_warning_type"] = None
            
            # Area for the alarm feature
            res["dim_alarms"] = cls.dim_alarms
        

            save_to_dict(
                res, cls, "alarms_area_names", (lambda li: [str(el) for el in li]), copy_
            )
            save_to_dict(
                res,
                cls,
                "alarms_lines_area",
                (
                    lambda dict_: {
                        str(l_nm): [str(ar_nm) for ar_nm in areas]
                        for l_nm, areas in dict_.items()
                    }
                ),
                copy_,
            )
            save_to_dict(
                res,
                cls,
                "alarms_area_lines",
                (lambda lili: [[str(l_nm) for l_nm in lines] for lines in lili]),
                copy_,
            )
            
            # No. of line alerst for the alert feature
            res['dim_alerts'] = cls.dim_alerts 
            # Save alert line names to dict
            save_to_dict(
                res, cls, "alertable_line_names", (lambda li: [str(el) for el in li]) if as_list else None, copy_
            )
            save_to_dict(
                res, cls, "alertable_line_ids", (lambda li: [int(el) for el in li])  if as_list else None, copy_
            )
            # Avoid further computation and save it
            if not as_list:
                cls._CLS_DICT = res.copy()
        return res

    @staticmethod
    def _make_cls_dict_extended(cls, res: CLS_AS_DICT_TYPING, as_list=True, copy_=True, _topo_vect_only=False):
        """add the n_gen and all in the class created
        
        Notes
        -------
        _topo_vect_only: this function is called once when the backend is initialized in `backend.load_grid`  
        (in `backend._compute_pos_big_topo`) and then once when everything is set up 
        (after redispatching and storage data are loaded).
        
        This is why I need the `_topo_vect_only` flag that tells this function when it's called only for 
        `topo_vect` related attributed
        
        """
        if cls._CLS_DICT_EXTENDED is not None and not as_list and not _topo_vect_only:
            # speed optimization: it has already been computed, so 
            # I reuse it (class attr are const)
            for k, v in cls._CLS_DICT_EXTENDED.items():
                if copy_:
                    res[k] = copy.deepcopy(v)
                else:
                    res[k] = v
            return
        
        GridObjects._make_cls_dict(cls, res, as_list=as_list, copy_=copy_, _topo_vect_only=_topo_vect_only)
        res["n_gen"] = cls.n_gen
        res["n_load"] = cls.n_load
        res["n_line"] = cls.n_line
        res["n_sub"] = cls.n_sub
        res["dim_topo"] = 1 * cls.dim_topo
        # storage
        res["n_storage"] = cls.n_storage
        # shunt (not in topo vect but might be usefull)
        res["shunts_data_available"] = cls.shunts_data_available
        res["n_shunt"] = cls.n_shunt
        
        if not _topo_vect_only:
            # all the attributes bellow are not needed for the "first call"
            # to this function when the elements are put together in the topo_vect.
            # Indeed, at this stage (first call in the backend.load_grid) these
            # attributes are not loaded yet
            
            # redispatching / curtailment
            res["redispatching_unit_commitment_available"] = cls.redispatching_unit_commitment_available

            # Flexible / redispatchable loads
            res["flexible_load_available"] = cls.flexible_load_available

            # n_busbar_per_sub
            res["n_busbar_per_sub"] = cls.n_busbar_per_sub
            
        # avoid further computation and save it
        if not as_list and not _topo_vect_only:
            cls._CLS_DICT_EXTENDED = res.copy()

    @classmethod
    def cls_to_dict(cls):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is used internally only to save action_space or observation_space for example. Do not
            attempt to use it in a different context.

        Convert the object as a dictionary.
        Note that unless this method is overridden, a call to it will only output the

        Returns
        -------
        res: ``dict``
            The representation of the object as a dictionary that can be json serializable.
        """
        res = {}
        cls._make_cls_dict(cls, res)
        return res

    @staticmethod
    def from_dict(dict_):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is used internally only to restore action_space or observation_space if they
            have been saved by `to_dict`. Do not
            attempt to use it in a different context.

        Create a valid GridObject (or one of its derived class if this method is overide) from a dictionnary (usually
        read from a json file)

        Parameters
        ----------
        dict_: ``dict``
            The representation of the GridObject as a dictionary.

        Returns
        -------
        res: :class:`GridObject`
            The object of the proper class that were initially represented as a dictionary.

        """
        # TODO refacto that with the "type(blablabla, blabla, blabal)" syntax !
        class res(GridObjects):
            pass

        cls = res
        if "glop_version" in dict_:
            cls.glop_version = str(dict_["glop_version"])
        else:
            cls.glop_version = cls.BEFORE_COMPAT_VERSION

        if "_PATH_GRID_CLASSES" in dict_:
            if dict_["_PATH_GRID_CLASSES"] is not None:
                cls._PATH_GRID_CLASSES = str(dict_["_PATH_GRID_CLASSES"])
            else:
                cls._PATH_GRID_CLASSES = None
        elif "_PATH_ENV" in dict_:
            # legacy mode in grid2op <= 1.10.1 this was saved in "PATH_ENV"
            if dict_["_PATH_ENV"] is not None:
                cls._PATH_GRID_CLASSES = str(dict_["_PATH_ENV"])
            else:
                cls._PATH_GRID_CLASSES = None
        else:
            cls._PATH_GRID_CLASSES = None
        
        if 'n_busbar_per_sub' in dict_:
            cls.n_busbar_per_sub = int(dict_["n_busbar_per_sub"])
        else:
            # compat version: was not set
            cls.n_busbar_per_sub = DEFAULT_N_BUSBAR_PER_SUB

        cls.name_gen = extract_from_dict(
            dict_, "name_gen", lambda x: np.array(x).astype(str)
        )
        cls.name_load = extract_from_dict(
            dict_, "name_load", lambda x: np.array(x).astype(str)
        )
        cls.name_line = extract_from_dict(
            dict_, "name_line", lambda x: np.array(x).astype(str)
        )
        cls.name_sub = extract_from_dict(
            dict_, "name_sub", lambda x: np.array(x).astype(str)
        )
        if "env_name" in dict_:
            # new saved in version >= 1.3.0
            cls.env_name = str(dict_["env_name"])
        else:
            # environment name was not stored, this make the task to retrieve this impossible
            pass

        cls.sub_info = extract_from_dict(
            dict_, "sub_info", lambda x: np.array(x).astype(dt_int)
        )
        cls.load_to_subid = extract_from_dict(
            dict_, "load_to_subid", lambda x: np.array(x).astype(dt_int)
        )
        cls.gen_to_subid = extract_from_dict(
            dict_, "gen_to_subid", lambda x: np.array(x).astype(dt_int)
        )
        cls.line_or_to_subid = extract_from_dict(
            dict_, "line_or_to_subid", lambda x: np.array(x).astype(dt_int)
        )
        cls.line_ex_to_subid = extract_from_dict(
            dict_, "line_ex_to_subid", lambda x: np.array(x).astype(dt_int)
        )
        cls.load_to_sub_pos = extract_from_dict(
            dict_, "load_to_sub_pos", lambda x: np.array(x).astype(dt_int)
        )
        cls.gen_to_sub_pos = extract_from_dict(
            dict_, "gen_to_sub_pos", lambda x: np.array(x).astype(dt_int)
        )
        cls.line_or_to_sub_pos = extract_from_dict(
            dict_, "line_or_to_sub_pos", lambda x: np.array(x).astype(dt_int)
        )
        cls.line_ex_to_sub_pos = extract_from_dict(
            dict_, "line_ex_to_sub_pos", lambda x: np.array(x).astype(dt_int)
        )
        cls.load_pos_topo_vect = extract_from_dict(
            dict_, "load_pos_topo_vect", lambda x: np.array(x).astype(dt_int)
        )
        cls.gen_pos_topo_vect = extract_from_dict(
            dict_, "gen_pos_topo_vect", lambda x: np.array(x).astype(dt_int)
        )
        cls.line_or_pos_topo_vect = extract_from_dict(
            dict_, "line_or_pos_topo_vect", lambda x: np.array(x).astype(dt_int)
        )
        cls.line_ex_pos_topo_vect = extract_from_dict(
            dict_, "line_ex_pos_topo_vect", lambda x: np.array(x).astype(dt_int)
        )

        cls.n_gen = len(cls.name_gen)
        cls.n_load = len(cls.name_load)
        cls.n_line = len(cls.name_line)
        cls.n_sub = len(cls.name_sub)
        cls.dim_topo = cls.sub_info.sum()

        if dict_["gen_type"] is None:
            cls.redispatching_unit_commitment_available = False
            # and no need to make anything else, because everything is already initialized at None
        else:
            cls.redispatching_unit_commitment_available = True
            type_attr_disp = [
                str,
                dt_float,
                dt_float,
                dt_bool,
                dt_float,
                dt_float,
                dt_int,
                dt_int,
                dt_float,
                dt_float,
                dt_float,
                dt_bool,
            ]

            # small "hack" here for the "gen_renewable" attribute, used for curtailment, that
            # is coded in grid2op >= 1.5 only
            if "gen_renewable" not in dict_:
                # before grid2op 1.4 it was not possible to make the difference between a renewable generator
                # and a non dispatchable one. Though no environment have this property yet, it is
                # possible to do it.
                dict_["gen_renewable"] = [not el for el in dict_["gen_redispatchable"]]

            for nm_attr, type_attr in zip(cls._li_attr_disp, type_attr_disp):
                setattr(
                    cls,
                    nm_attr,
                    extract_from_dict(
                        dict_, nm_attr, lambda x: np.array(x).astype(type_attr)
                    ),
                )
        
        cls.flexible_load_available = False
        set_flex_defaults  = True
        if "load_flexible" in dict_:
            if dict_["load_flexible"] is not None:
                cls.flexible_load_available = True
                set_flex_defaults = False
        # Enables backwards compatibility with Flexibility (introduced 1.10.4)
        GridObjects.set_cls_flexibility(dict_, set_defaults=set_flex_defaults)

            
        cls.grid_layout = extract_from_dict(dict_, "grid_layout", lambda x: x)
        cls.name_shunt = extract_from_dict(dict_, "name_shunt", lambda x: x)
        if cls.name_shunt is not None:
            cls.shunts_data_available = True
            cls.n_shunt = len(cls.name_shunt)
            cls.name_shunt = np.array(cls.name_shunt).astype(str)
            cls.shunt_to_subid = extract_from_dict(
                dict_, "shunt_to_subid", lambda x: np.array(x).astype(dt_int)
            )

        if "name_storage" in dict_:
            # this is for backward compatibility with logs coming from grid2op <= 1.5
            # where storage unit did not exist.
            cls.name_storage = extract_from_dict(
                dict_, "name_storage", lambda x: np.array(x).astype(str)
            )
            cls.storage_to_subid = extract_from_dict(
                dict_, "storage_to_subid", lambda x: np.array(x).astype(dt_int)
            )
            cls.storage_to_sub_pos = extract_from_dict(
                dict_, "storage_to_sub_pos", lambda x: np.array(x).astype(dt_int)
            )
            cls.storage_pos_topo_vect = extract_from_dict(
                dict_, "storage_pos_topo_vect", lambda x: np.array(x).astype(dt_int)
            )
            cls.n_storage = len(cls.name_storage)
            
            # storage static data
            cls.storage_type = extract_from_dict(dict_, "storage_type", lambda x: np.array(x).astype(str))
            cls.storage_Emax = extract_from_dict(
                dict_, "storage_Emax", lambda x: np.array(x).astype(dt_float)
            )
            cls.storage_Emin = extract_from_dict(
                dict_, "storage_Emin", lambda x: np.array(x).astype(dt_float)
            )
            cls.storage_max_p_prod = extract_from_dict(
                dict_, "storage_max_p_prod", lambda x: np.array(x).astype(dt_float)
            )
            cls.storage_max_p_absorb = extract_from_dict(
                dict_, "storage_max_p_absorb", lambda x: np.array(x).astype(dt_float)
            )
            cls.storage_marginal_cost = extract_from_dict(
                dict_, "storage_marginal_cost", lambda x: np.array(x).astype(dt_float)
            )
            cls.storage_loss = extract_from_dict(
                dict_, "storage_loss", lambda x: np.array(x).astype(dt_float)
            )
            cls.storage_charging_efficiency = extract_from_dict(
                dict_,
                "storage_charging_efficiency",
                lambda x: np.array(x).astype(dt_float),
            )
            cls.storage_discharging_efficiency = extract_from_dict(
                dict_,
                "storage_discharging_efficiency",
                lambda x: np.array(x).astype(dt_float),
            )
        else:
            # backward compatibility: no storage were supported
            cls.set_no_storage()
            
        cls.process_shunt_satic_data()
        
        if cls.glop_version != grid2op.__version__:
            # change name of the environment, this is done in Environment.py for regular environment
            # see `self.backend.set_env_name(f"{self.name}_{self._compat_glop_version}")`
            # cls.set_env_name(f"{cls.env_name}_{cls.glop_version}")
            # and now post process the class attributes for that
            cls.process_grid2op_compat()

        if "assistant_warning_type" in dict_:
            cls.assistant_warning_type = dict_["assistant_warning_type"]
        else:
            cls.assistant_warning_type = None
        
        # alarm information
        if "dim_alarms" in dict_:
            # NB by default the constructor do as if there were no alarm so that's great !
            cls.dim_alarms = dict_["dim_alarms"]
            cls.alarms_area_names = copy.deepcopy(dict_["alarms_area_names"])
            cls.alarms_lines_area = copy.deepcopy(dict_["alarms_lines_area"])
            cls.alarms_area_lines = copy.deepcopy(dict_["alarms_area_lines"])

        # alert information
        if "dim_alerts" in dict_:
            # NB by default the constructor do as if there were no alert so that's great !
            cls.dim_alerts = dict_["dim_alerts"]
            if cls.dim_alerts > 0:
                cls.alertable_line_names = extract_from_dict(
                    dict_, "alertable_line_names", lambda x: np.array(x).astype(str)
                    )
                cls.alertable_line_ids = extract_from_dict(
                    dict_, "alertable_line_ids", lambda x: np.array(x).astype(dt_int)
                    )
            else:
                cls.alertable_line_names = []
                cls.alertable_line_ids = []
        
        # save the representation of this class as dict
        tmp = {}
        cls._make_cls_dict_extended(cls, tmp, as_list=False, copy_=True)  
        
        # retrieve the redundant information that are not stored (for efficiency)
        obj_ = cls()
        obj_._compute_pos_big_topo_cls()
        cls = cls.init_grid(obj_)  # , force=True
        return cls()
    
    @classmethod
    def set_cls_flexibility(cls, dict_:dict, set_defaults:bool=True) -> dict:
        """
        Set flexibility-related attributes for GridObjects. Can use
        default-values for Backwards compatability.

        Args:
            dict_ (dict): Dictionary from which 'GridObjects' is constructed
            set_defaults (bool, optional): Whether to fill with default-values. Defaults to True.

        Returns:
            dict: Modified Dictionary from which 'GridObjects' is constructed
        """
        # Enables backwards compatibility with Flexibility (introduced 1.10.4)
        type_attr_flex_load = [dt_float, dt_bool, dt_float,
                            dt_float, dt_int, dt_int, dt_float]
        if set_defaults:
            prim_neutral_lookup = {bool:False, float:0.0, int:0, str:""}
            for attr_name, attr_type in zip(cls._li_attr_flex_load, cls._type_attr_flex_load):
                dict_[attr_name] = [prim_neutral_lookup[attr_type]]*cls.n_load
        for nm_attr, type_attr in zip(cls._li_attr_flex_load, type_attr_flex_load):
            setattr(cls, nm_attr, extract_from_dict(dict_, nm_attr,
                    lambda x: np.array(x).astype(type_attr)))
        return dict_

    @classmethod
    def process_shunt_satic_data(cls):
        """remove possible shunts data from the classes, if shunts are deactivated"""
        pass
    
    @classmethod
    def set_no_storage(cls):
        """
        this function is used to set all necessary parameters when the grid do not contain any storage element.

        Returns
        -------

        """
        GridObjects.deactivate_storage(cls)

    @staticmethod
    def deactivate_storage(obj):
        obj.n_storage = 0
        obj.name_storage = np.array([], dtype=str)
        obj.storage_to_subid = np.array([], dtype=dt_int)
        obj.storage_pos_topo_vect = np.array([], dtype=dt_int)
        obj.storage_to_sub_pos = np.array([], dtype=dt_int)

        obj.storage_type = np.array([], dtype=str)
        obj.storage_Emax = np.array([], dtype=dt_float)
        obj.storage_Emin = np.array([], dtype=dt_float)
        obj.storage_max_p_prod = np.array([], dtype=dt_float)
        obj.storage_max_p_absorb = np.array([], dtype=dt_float)
        obj.storage_marginal_cost = np.array([], dtype=dt_float)
        obj.storage_loss = np.array([], dtype=dt_float)
        obj.storage_charging_efficiency = np.array([], dtype=dt_float)
        obj.storage_discharging_efficiency = np.array([], dtype=dt_float)

    @classmethod
    def same_grid_class(cls, other_cls) -> bool:
        """
        return whether the two classes have the same grid

        Notes
        ------
        Two environments can have different name, but representing the same grid. This is why this function
        is agnostic to the "env_name" class attribute.

        In order for two grid to be equal, they must have everything in common, including the presence /
        absence of shunts or storage units for example.

        """
        if cls.env_name == other_cls.env_name:
            # speed optimization here: if the two classes are from the same environment
            # they are from the same grid !
            return True

        # this implementation is 6 times faster than the "cls_to_dict" one below, so i kept it
        me_dict = {}
        GridObjects._make_cls_dict_extended(cls, me_dict, as_list=False, copy_=False)
        other_cls_dict = {}
        GridObjects._make_cls_dict_extended(other_cls, other_cls_dict, as_list=False, copy_=False) 

        if me_dict.keys() - other_cls_dict.keys():
            # one key is in me but not in other
            return False
        if other_cls_dict.keys() - me_dict.keys():
            # one key is in other but not in me
            return False
        for attr_nm in me_dict.keys():
            if attr_nm == "env_name":
                continue
            if attr_nm.startswith("__") and attr_nm.endswith("__"):
                continue
            if not np.array_equal(getattr(cls, attr_nm), getattr(other_cls, attr_nm)):
                return False
        return True

    @staticmethod
    def _build_cls_from_import(name_cls, path_env):
        import sys
        import os
        import importlib

        my_class = None
        if path_env is None:
            return None
        if not os.path.exists(path_env):
            return None
        if not os.path.isdir(path_env):
            return None
        if not os.path.exists(os.path.join(path_env, "_grid2op_classes")):
            return None
        sys.path.append(path_env)
        try:
            module = importlib.import_module("_grid2op_classes")
            if hasattr(module, name_cls):
                my_class = getattr(module, name_cls)
        except (ModuleNotFoundError, ImportError) as exc_:
            # normal behaviour i don't do anything there
            # TODO explain why
            pass
        return my_class

    @staticmethod
    def init_grid_from_dict_for_pickle(name_res, orig_cls, cls_attr):
        """
        This function is used internally for pickle to build the classes of the
        objects instead of loading them from the module (which is impossible as
        most classes are defined on the fly in grid2op)

        It is expected to create an object of the correct type. This object will then be
        "filled" with the proper content automatically by python, because i provided the "state" of the
        object in the __reduce__ method.
        """
        res_cls = None
        if "_PATH_GRID_CLASSES" in cls_attr and cls_attr["_PATH_GRID_CLASSES"] is not None:
            res_cls = GridObjects._build_cls_from_import(
                name_res, cls_attr["_PATH_GRID_CLASSES"]
            )

        # check if the class already exists, if so returns it
        if res_cls is not None:
            # i recreate the class from local import
            pass
        elif name_res in globals():
            # no need to recreate the class, it already exists
            res_cls = globals()[name_res]
        else:
            # define properly the class, as it is not found
            res_cls = type(name_res, (orig_cls,), cls_attr)
            res_cls._INIT_GRID_CLS = orig_cls  # don't forget to remember the base class
            # if hasattr(res_cls, "n_sub") and res_cls.n_sub > 0:
            # that's a grid2op class iniailized with an environment, I need to initialize it too
            res_cls._compute_pos_big_topo_cls()
            if res_cls.glop_version != grid2op.__version__:
                res_cls.process_grid2op_compat()
            res_cls.process_shunt_satic_data()
            # add the class in the "globals" for reuse later
            globals()[name_res] = res_cls

        # now create an "empty" object (using new)
        res = res_cls.__new__(res_cls)
        return res

    # used for pickle and for deep copy
    def __reduce__(self):
        """
        It here to avoid issue with pickle.
        But the problem is that it's also used by deepcopy... So its implementation is used a lot
        
        see https://docs.python.org/3/library/pickle.html#object.__reduce__
        """
        # TODO this is not really a convenient use of that i'm sure !
        # Try to see if it can be better
        cls_attr_as_dict = {}
        GridObjects._make_cls_dict_extended(type(self), cls_attr_as_dict, as_list=False)  # TODO save that in the class definition
        if hasattr(self, "__getstate__"):
            my_state = self.__getstate__()
        else:
            my_state = {}
            for k, v in self.__dict__.items():
                my_state[k] = v  # copy.copy(v)

        my_cls = type(self)
        if hasattr(my_cls, "_INIT_GRID_CLS"):
            # I am a type created when an environment is loaded
            base_cls = my_cls._INIT_GRID_CLS
        else:
            # i am a "raw" type directly coming from grid2op
            base_cls = my_cls
        return (
            GridObjects.init_grid_from_dict_for_pickle,
            (type(self).__name__, base_cls, cls_attr_as_dict),
            my_state,
        )

    @classmethod
    def local_bus_to_global(cls, local_bus: np.ndarray, to_sub_id: np.ndarray) -> np.ndarray:
        """This function translate "local bus" whose id are in a substation, to "global bus id" whose
        id are consistent for the whole grid.
        
        Be carefull, when using this function, you might end up with deactivated bus: *eg* if you have an element on bus 
        with global id 1 and another on bus with global id 42 you might not have any element on bus with
        global id 41 or 40 or 39 or etc.
        
        .. note::
            Typically, "local bus" are numbered 1, 2, ... cls.n_busbar_per_sub. They represent the id of the busbar to which the element
            is connected IN its substation.
            
            On the other hand, the "global bus" are numberd, 0, 1, 2, 3, ..., 2 * self.n_sub. They represent some kind of 
            "universal" labelling of the busbars of all the grid. For example, substation 0 might have busbar `0` and `self.n_sub`, 
            substation 1 have busbar `1` and `self.n_sub + 1` etc.
            
            Local and global bus id represents the same thing. The difference comes down to convention.
            
        ..warning::
            In order to be as fast as possible, these functions do not check for "out of bound" or
            "impossible" configuration. 
            
            They assume that the input data are consistent with the grid.
        """
        global_bus = (1 * local_bus).astype(dt_int)  # make a copy
        global_bus[local_bus < 0] = -1
        for i in range(cls.n_busbar_per_sub):
            on_bus_i = local_bus == i + 1
            global_bus[on_bus_i] = to_sub_id[on_bus_i] + i * cls.n_sub
        return global_bus

    @classmethod
    def local_bus_to_global_int(cls, local_bus : int, to_sub_id : int) -> int:
        """This function translate "local bus" whose id are in a substation, to "global bus id" whose
        id are consistent for the whole grid.
        
        Be carefull, when using this function, you might end up with deactivated bus: *eg* if you have an element on bus 
        with global id 1 and another on bus with global id 42 you might not have any element on bus with
        global id 41 or 40 or 39 or etc.
        
        .. note::
            Typically, "local bus" are numbered 1, 2, ... cls.n_busbar_per_sub. They represent the id of the busbar to which the element
            is connected IN its substation.
            
            On the other hand, the "global bus" are numberd, 0, 1, 2, 3, ..., cls.n_busbar_per_sub  * self.n_sub. They represent some kind of 
            "universal" labelling of the busbars of all the grid. For example, substation 0 might have busbar `0` and `self.n_sub`, 
            substation 1 have busbar `1` and `self.n_sub + 1` etc.
            
            Local and global bus id represents the same thing. The difference comes down to convention.
            
        .. note::
            This is the "non vectorized" version that applies only on integers.          
              
        ..warning::
            In order to be as fast as possible, these functions do not check for "out of bound" or
            "impossible" configuration. 
            
            They assume that the input data are consistent with the grid.
        """
        if local_bus == -1:
            return -1
        return to_sub_id + (int(local_bus) - 1) * cls.n_sub

    @classmethod
    def global_bus_to_local(cls, global_bus: np.ndarray, to_sub_id: np.ndarray) -> np.ndarray:
        """This function translate "local bus" whose id are in a substation, to "global bus id" whose
        id are consistent for the whole grid.
        
        Be carefull, when using this function, you might end up with deactivated bus: *eg* if you have an element on bus 
        with global id 1 and another on bus with global id 42 you might not have any element on bus with
        global id 41 or 40 or 39 or etc.
        
        .. note::
            Typically, "local bus" are numbered 1, 2, ... cls.n_busbar_per_sub. They represent the id of the busbar to which the element
            is connected IN its substation.
            
            On the other hand, the "global bus" are numberd, 0, 1, 2, 3, ..., cls.n_busbar_per_sub * self.n_sub. They represent some kind of 
            "universal" labelling of the busbars of all the grid. For example, substation 0 might have busbar `0` and `self.n_sub`, 
            substation 1 have busbar `1` and `self.n_sub + 1` etc.
            
            Local and global bus id represents the same thing. The difference comes down to convention.
                        
        ..warning::
            In order to be as fast as possible, these functions do not check for "out of bound" or
            "impossible" configuration. 
            
            They assume that the input data are consistent with the grid.
        """
        res = (1 * global_bus).astype(dt_int)  # make a copy
        for i in range(cls.n_busbar_per_sub):
            res[(i * cls.n_sub <= global_bus) & (global_bus < (i+1) * cls.n_sub)] = i + 1
        res[global_bus == -1] = -1
        return res
    
    @classmethod
    def global_bus_to_local_int(cls, global_bus: int, to_sub_id: int) -> int:
        """This function translate "local bus" whose id are in a substation, to "global bus id" whose
        id are consistent for the whole grid.
        
        Be carefull, when using this function, you might end up with deactivated bus: *eg* if you have an element on bus 
        with global id 1 and another on bus with global id 42 you might not have any element on bus with
        global id 41 or 40 or 39 or etc.
        
        .. note::
            Typically, "local bus" are numbered 1, 2, ... cls.n_busbar_per_sub. They represent the id of the busbar to which the element
            is connected IN its substation.
            
            On the other hand, the "global bus" are numberd, 0, 1, 2, 3, ..., cls.n_busbar_per_sub * self.n_sub. They represent some kind of 
            "universal" labelling of the busbars of all the grid. For example, substation 0 might have busbar `0` and `self.n_sub`, 
            substation 1 have busbar `1` and `self.n_sub + 1` etc.
            
            Local and global bus id represents the same thing. The difference comes down to convention.      
                  
        ..warning::
            In order to be as fast as possible, these functions do not check for "out of bound" or
            "impossible" configuration. 
            
            They assume that the input data are consistent with the grid.
        """
        if global_bus == -1:
            return -1
        for i in range(cls.n_busbar_per_sub):
            if global_bus < (i+1) * cls.n_sub:
                return i+1
        raise EnvError(f"This environment can have only {cls.n_busbar_per_sub} independant busbars per substation.")
    
    @staticmethod
    def _format_int_vect_to_cls_str(int_vect):
        int_vect_str = "None"
        if int_vect is not None:
            int_vect_str = ",".join([f"{el}" for el in int_vect])
            int_vect_str = f"np.array([{int_vect_str}], dtype=dt_int)"
        return int_vect_str

    @staticmethod
    def _format_float_vect_to_cls_str(float_vect):
        float_vect_str = "None"
        if float_vect is not None:
            float_vect_str = ",".join([f"{el}" for el in float_vect])
            float_vect_str = f"np.array([{float_vect_str}], dtype=dt_float)"
        return float_vect_str

    @staticmethod
    def _format_bool_vect_to_cls_str(bool_vect):
        bool_vect_str = "None"
        if bool_vect is not None:
            bool_vect_str = ",".join(["True" if el else "False" for el in bool_vect])
            bool_vect_str = f"np.array([{bool_vect_str}], dtype=dt_bool)"
        return bool_vect_str

    @classmethod
    def _get_full_cls_str(cls):
        _PATH_ENV_str = "None" if cls._PATH_GRID_CLASSES is None else f'"{cls._PATH_GRID_CLASSES}"'
        attr_list_vect_str = None
        attr_list_set_str = "{}"
        if cls.attr_list_vect is not None:
            attr_list_vect_str = f"{cls.attr_list_vect}"
            attr_list_set_str = "set(attr_list_vect)"

        attr_list_json_str = None
        if cls.attr_list_json is not None:
            attr_list_json_str = f"{cls.attr_list_json}"

        attr_nan_list_set_str = "{}"
        if cls.attr_nan_list_set is not None:
            tmp_ = ",".join([f'"{el}"' for el in sorted(cls.attr_nan_list_set)])
            attr_nan_list_set_str = f"set([{tmp_}])"

        name_load_str = ",".join([f'"{el}"' for el in cls.name_load])
        name_gen_str = ",".join([f'"{el}"' for el in cls.name_gen])
        name_line_str = ",".join([f'"{el}"' for el in cls.name_line])
        name_storage_str = ",".join([f'"{el}"' for el in cls.name_storage])
        name_sub_str = ",".join([f'"{el}"' for el in cls.name_sub])

        sub_info_str = GridObjects._format_int_vect_to_cls_str(cls.sub_info)

        load_to_subid_str = GridObjects._format_int_vect_to_cls_str(cls.load_to_subid)
        gen_to_subid_str = GridObjects._format_int_vect_to_cls_str(cls.gen_to_subid)
        line_or_to_subid_str = GridObjects._format_int_vect_to_cls_str(
            cls.line_or_to_subid
        )
        line_ex_to_subid_str = GridObjects._format_int_vect_to_cls_str(
            cls.line_ex_to_subid
        )
        storage_to_subid_str = GridObjects._format_int_vect_to_cls_str(
            cls.storage_to_subid
        )

        # which index has this element in the substation vector
        load_to_sub_pos_str = GridObjects._format_int_vect_to_cls_str(
            cls.load_to_sub_pos
        )
        gen_to_sub_pos_str = GridObjects._format_int_vect_to_cls_str(cls.gen_to_sub_pos)
        line_or_to_sub_pos_str = GridObjects._format_int_vect_to_cls_str(
            cls.line_or_to_sub_pos
        )
        line_ex_to_sub_pos_str = GridObjects._format_int_vect_to_cls_str(
            cls.line_ex_to_sub_pos
        )
        storage_to_sub_pos_str = GridObjects._format_int_vect_to_cls_str(
            cls.storage_to_sub_pos
        )

        # which index has this element in the topology vector
        load_pos_topo_vect_str = GridObjects._format_int_vect_to_cls_str(
            cls.load_pos_topo_vect
        )
        gen_pos_topo_vect_str = GridObjects._format_int_vect_to_cls_str(
            cls.gen_pos_topo_vect
        )
        line_or_pos_topo_vect_str = GridObjects._format_int_vect_to_cls_str(
            cls.line_or_pos_topo_vect
        )
        line_ex_pos_topo_vect_str = GridObjects._format_int_vect_to_cls_str(
            cls.line_ex_pos_topo_vect
        )
        storage_pos_topo_vect_str = GridObjects._format_int_vect_to_cls_str(
            cls.storage_pos_topo_vect
        )

        def format_el_int(values):
            return ",".join([f"{el}" for el in values])

        tmp_tmp_ = [format_el_int(el) for el in cls.grid_objects_types]
        tmp_ = ",".join([f"[{el}]" for el in tmp_tmp_])
        grid_objects_types_str = f"np.array([{tmp_}], " f"dtype=dt_int)"
        _topo_vect_to_sub_str = GridObjects._format_int_vect_to_cls_str(
            cls._topo_vect_to_sub
        )

        _vectorized_str = "None"

        gen_type_str = (
            ",".join([f'"{el}"' for el in cls.gen_type])
            if cls.redispatching_unit_commitment_available
            else "None"
        )
        gen_pmin_str = GridObjects._format_float_vect_to_cls_str(cls.gen_pmin)
        gen_pmax_str = GridObjects._format_float_vect_to_cls_str(cls.gen_pmax)
        gen_redispatchable_str = GridObjects._format_bool_vect_to_cls_str(
            cls.gen_redispatchable
        )
        gen_max_ramp_up_str = GridObjects._format_float_vect_to_cls_str(
            cls.gen_max_ramp_up
        )
        gen_max_ramp_down_str = GridObjects._format_float_vect_to_cls_str(
            cls.gen_max_ramp_down
        )
        gen_min_uptime_str = GridObjects._format_int_vect_to_cls_str(cls.gen_min_uptime)
        gen_min_downtime_str = GridObjects._format_int_vect_to_cls_str(
            cls.gen_min_downtime
        )
        gen_cost_per_MW_str = GridObjects._format_float_vect_to_cls_str(
            cls.gen_cost_per_MW
        )
        gen_startup_cost_str = GridObjects._format_float_vect_to_cls_str(
            cls.gen_startup_cost
        )
        gen_shutdown_cost_str = GridObjects._format_float_vect_to_cls_str(
            cls.gen_shutdown_cost
        )
        gen_renewable_str = GridObjects._format_bool_vect_to_cls_str(cls.gen_renewable)

        load_size_str = GridObjects._format_float_vect_to_cls_str(cls.load_size)
        load_flexible_str = GridObjects._format_bool_vect_to_cls_str(cls.load_flexible)
        load_max_ramp_up_str = GridObjects._format_float_vect_to_cls_str(cls.load_max_ramp_up)
        load_max_ramp_down_str = GridObjects._format_float_vect_to_cls_str(cls.load_max_ramp_down)
        load_min_uptime_str = GridObjects._format_int_vect_to_cls_str(cls.load_min_uptime)
        load_min_downtime_str = GridObjects._format_int_vect_to_cls_str(cls.load_min_downtime)
        load_cost_per_MW_str = GridObjects._format_float_vect_to_cls_str(cls.load_cost_per_MW)

        storage_type_str = ",".join([f'"{el}"' for el in cls.storage_type])
        storage_Emax_str = GridObjects._format_float_vect_to_cls_str(cls.storage_Emax)
        storage_Emin_str = GridObjects._format_float_vect_to_cls_str(cls.storage_Emin)
        storage_max_p_prod_str = GridObjects._format_float_vect_to_cls_str(
            cls.storage_max_p_prod
        )
        storage_max_p_absorb_str = GridObjects._format_float_vect_to_cls_str(
            cls.storage_max_p_absorb
        )
        storage_marginal_cost_str = GridObjects._format_float_vect_to_cls_str(
            cls.storage_marginal_cost
        )
        storage_loss_str = GridObjects._format_float_vect_to_cls_str(cls.storage_loss)
        storage_charging_efficiency_str = GridObjects._format_float_vect_to_cls_str(
            cls.storage_charging_efficiency
        )
        storage_discharging_efficiency_str = GridObjects._format_float_vect_to_cls_str(
            cls.storage_discharging_efficiency
        )

        def format_el(values):
            return ",".join([f'"{el}"' for el in values])

        if cls.grid_layout is not None:
            tmp_tmp_ = [f'"{k}": [{format_el(v)}]' for k, v in cls.grid_layout.items()]
            tmp_ = ",".join(tmp_tmp_)
            grid_layout_str = f"{{{tmp_}}}"
        else:
            grid_layout_str = "None"

        name_shunt_str = ",".join([f'"{el}"' for el in cls.name_shunt])
        shunt_to_subid_str = GridObjects._format_int_vect_to_cls_str(cls.shunt_to_subid)

        assistant_warning_type_str = (None if cls.assistant_warning_type is None 
                                      else f'"{cls.assistant_warning_type}"')
        alarms_area_names_str = (
            "[]"
            if cls.dim_alarms == 0
            else ",".join([f'"{el}"' for el in cls.alarms_area_names])
        )

        tmp_tmp_ = ",".join(
            [f'"{k}": [{format_el(v)}]' for k, v in cls.alarms_lines_area.items()]
        )
        tmp_ = f"{{{tmp_tmp_}}}"
        alarms_lines_area_str = "{}" if cls.dim_alarms == 0 else tmp_

        tmp_tmp_ = ",".join([f"[{format_el(el)}]" for el in cls.alarms_area_lines])
        tmp_ = f"[{tmp_tmp_}]"
        alarms_area_lines_str = "[]" if cls.dim_alarms == 0 else tmp_

        tmp_tmp_ = ",".join([f"\"{el}\"" for el in cls.alertable_line_names])
        tmp_ = f"[{tmp_tmp_}]"
        alertable_line_names_str = '[]' if cls.dim_alerts == 0 else tmp_
        
        tmp_tmp_ = ",".join([f"{el}" for el in cls.alertable_line_ids])
        tmp_ = f"[{tmp_tmp_}]"
        alertable_line_ids_str = '[]' if cls.dim_alerts == 0 else tmp_
        res = f"""# Copyright (c) 2019-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

# THIS FILE HAS BEEN AUTOMATICALLY GENERATED BY "gridobject._get_full_cls_str()"
# WE DO NOT RECOMMEND TO ALTER IT IN ANY WAY
import numpy as np

import grid2op
from grid2op.dtypes import dt_int, dt_float, dt_bool
from {cls._INIT_GRID_CLS.__module__} import {cls._INIT_GRID_CLS.__name__}


class {cls.__name__}({cls._INIT_GRID_CLS.__name__}):
    BEFORE_COMPAT_VERSION = \"{cls.BEFORE_COMPAT_VERSION}\"
    glop_version = grid2op.__version__  # tells it's the installed grid2op version
    _PATH_GRID_CLASSES = {_PATH_ENV_str}   # especially do not modify that
    _INIT_GRID_CLS = {cls._INIT_GRID_CLS.__name__} 
    _CLS_DICT = None  # init once to avoid yet another serialization of the class as dict (in make_cls_dict)
    _CLS_DICT_EXTENDED = None  # init once to avoid yet another serialization of the class as dict  (in make_cls_dict)

    SUB_COL = 0
    LOA_COL = 1
    GEN_COL = 2
    LOR_COL = 3
    LEX_COL = 4
    STORAGE_COL = 5

    attr_list_vect = {attr_list_vect_str}
    attr_list_set = {attr_list_set_str}
    attr_list_json = {attr_list_json_str}
    attr_nan_list_set = {attr_nan_list_set_str}

    # name of the objects
    env_name = "{cls.env_name}"
    name_load = np.array([{name_load_str}], dtype=str)
    name_gen = np.array([{name_gen_str}], dtype=str)
    name_line = np.array([{name_line_str}], dtype=str)
    name_sub = np.array([{name_sub_str}], dtype=str)
    name_storage = np.array([{name_storage_str}], dtype=str)

    n_busbar_per_sub = {cls.n_busbar_per_sub}
    n_gen = {cls.n_gen}
    n_load = {cls.n_load}
    n_line = {cls.n_line}
    n_sub = {cls.n_sub}
    n_storage = {cls.n_storage}

    sub_info = {sub_info_str}
    dim_topo = {cls.dim_topo}

    # to which substation is connected each element
    load_to_subid = {load_to_subid_str}
    gen_to_subid = {gen_to_subid_str}
    line_or_to_subid = {line_or_to_subid_str}
    line_ex_to_subid = {line_ex_to_subid_str}
    storage_to_subid = {storage_to_subid_str}

    # which index has this element in the substation vector
    load_to_sub_pos = {load_to_sub_pos_str}
    gen_to_sub_pos = {gen_to_sub_pos_str}
    line_or_to_sub_pos = {line_or_to_sub_pos_str}
    line_ex_to_sub_pos = {line_ex_to_sub_pos_str}
    storage_to_sub_pos = {storage_to_sub_pos_str}

    # which index has this element in the topology vector
    load_pos_topo_vect = {load_pos_topo_vect_str}
    gen_pos_topo_vect = {gen_pos_topo_vect_str}
    line_or_pos_topo_vect = {line_or_pos_topo_vect_str}
    line_ex_pos_topo_vect = {line_ex_pos_topo_vect_str}
    storage_pos_topo_vect = {storage_pos_topo_vect_str}

    # "convenient" way to retrieve information of the grid
    grid_objects_types = {grid_objects_types_str}
    # to which substation each element of the topovect is connected
    _topo_vect_to_sub = {_topo_vect_to_sub_str}

    # list of attribute to convert it from/to a vector
    _vectorized = {_vectorized_str}

    # for redispatching / unit commitment
    _li_attr_disp = ["gen_type", "gen_pmin", "gen_pmax", "gen_redispatchable", "gen_max_ramp_up",
                     "gen_max_ramp_down", "gen_min_uptime", "gen_min_downtime", "gen_cost_per_MW",
                     "gen_startup_cost", "gen_shutdown_cost", "gen_renewable"]

    _type_attr_disp = [str, float, float, bool, float, float, int, int, float, float, float, bool]

    # redispatch data, not available in all environment
    redispatching_unit_commitment_available = {"True" if cls.redispatching_unit_commitment_available else "False"}
    gen_type = np.array([{gen_type_str}])
    gen_pmin = {gen_pmin_str}
    gen_pmax = {gen_pmax_str}
    gen_redispatchable = {gen_redispatchable_str}
    gen_max_ramp_up = {gen_max_ramp_up_str}
    gen_max_ramp_down = {gen_max_ramp_down_str}
    gen_min_uptime = {gen_min_uptime_str}
    gen_min_downtime = {gen_min_downtime_str}
    gen_cost_per_MW = {gen_cost_per_MW_str}  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )
    gen_startup_cost = {gen_startup_cost_str}  # start cost (in currency)
    gen_shutdown_cost = {gen_shutdown_cost_str}  # shutdown cost (in currency)
    gen_renewable = {gen_renewable_str}

    # for flexibility
    _li_attr_flex = ["load_size", "load_flexible", "load_max_ramp_up",
                     "load_max_ramp_down", "load_min_uptime", "load_min_downtime", "load_cost_per_MW"]

    _type_attr_flex = [float, float, bool, float, float, int, int, float]

    # Flexible load data, not available in all environments
    flexible_load_available = {"True" if cls.flexible_load_available else "False"}
    load_size = {load_size_str}
    load_flexible = {load_flexible_str}
    load_max_ramp_up = {load_max_ramp_up_str}
    load_max_ramp_down = {load_max_ramp_down_str}
    load_min_uptime = {load_min_uptime_str}
    load_min_downtime = {load_min_downtime_str}
    load_cost_per_MW = {load_cost_per_MW_str}  # marginal cost (in currency / (power.step) and not in $/(MW.h) it would be $ / (MW.5mins) )

    # storage unit static data
    storage_type = np.array([{storage_type_str}], dtype=str)
    storage_Emax = {storage_Emax_str}
    storage_Emin = {storage_Emin_str}
    storage_max_p_prod = {storage_max_p_prod_str}
    storage_max_p_absorb = {storage_max_p_absorb_str}
    storage_marginal_cost = {storage_marginal_cost_str}
    storage_loss = {storage_loss_str}
    storage_charging_efficiency = {storage_charging_efficiency_str}
    storage_discharging_efficiency = {storage_discharging_efficiency_str}

    # grid layout
    grid_layout = {grid_layout_str}

    # shunt data, not available in every backend
    shunts_data_available = {"True" if cls.shunts_data_available else "False"}
    n_shunt = {cls.n_shunt}
    name_shunt = np.array([{name_shunt_str}])
    shunt_to_subid = {shunt_to_subid_str}

    # alarm / alert
    assistant_warning_type = {assistant_warning_type_str}
    
    # alarm feature
    # dimension of the alarm "space" (number of alarm that can be raised at each step)
    dim_alarms = {cls.dim_alarms}
    alarms_area_names = {alarms_area_names_str}
    alarms_lines_area = {alarms_lines_area_str}
    alarms_area_lines = {alarms_area_lines_str}

    # alert feature
    dim_alerts = {cls.dim_alerts}
    alertable_line_names = {alertable_line_names_str}
    alertable_line_ids = {alertable_line_ids_str}

"""
        return res
