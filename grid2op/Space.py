"""
This class abstracts the main components of Action, Observation, ActionSpace, and ObservationSpace.

It represents a powergrid (the object in it) in a format completely agnostic to the solver used to compute
the power flows (:class:`grid2op.Backend.Backend`).

"""
import re
import json
import os
import copy

import numpy as np

try:
    from .Exceptions import *
    from ._utils import extract_from_dict, save_to_dict
except (ModuleNotFoundError, ImportError):
    from Exceptions import *
    from _utils import extract_from_dict, save_to_dict

import pdb


# TODO better random stuff when random observation (seed in argument is really weird)

# TODO tests of these methods and this class in general

class GridObjects:
    """
    This class stores in a Backend agnostic way some information about the powergrid.

    It stores information about numbers of objects, and which objects are where, their names, etc.

    The classes :class:`grid2op.Action.Action`, :class:`grid2op.Action.HelperAction`,
    :class:`grid2op.Observation.Observation`, :class:`grid2op.Observation.ObservationHelper` and
    :class:`grid2op.Backend.Backend` all inherit from this class. This means that each of the above has its own
    representation of the powergrid.


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
    position of :attr:`GridObjects.gen_to_sub_pos` indicates on which component of the (fictive) vector representing the
    substation 1 to look to know on which bus the first generator is connected.

    We define the "topology" as the busbar to which each object is connected: each object being connected to either
    busbar 1 or busbar 2, this topology can be represented by a vector of fixed size (and it actually is in
    :attr:`grid2op.Observation.Observation.topo_vect` or in :func:`grid2op.Backend.Backend.get_topo_vect`). There are
    multiple ways to make such a vector. We decided to concatenate all the (fictive) vectors described above. This
    concatenation represents the actual topology of this powergrid at a given timestep. This class doesn't store this
    information (see :class:`grid2op.Observation.Observation` for such purpose).
    This entails that:

    - the bus to which each object on a substation will be stored in consecutive components of such a vector. For
      example, if the first substation of the grid has 5 elements connected to it, then the first 5 elements of
      :attr:`grid2op.Observation.Observation.topo_vect` will represent these 5 elements. The number of elements
      in each substation is given in :attr:`grid2op.Space.GridObjects.sub_info`.
    - the substation are stored in "order": objects of the first substations are represented, then this is the objects
      of the second substation etc. So in the example above, the 6th element of
      :attr:`grid2op.Observation.Observation.topo_vect` is an object connected to the second substation.
    - to know on which position of this "topology vector" we can find the information relative a specific element
      it is possible to:

        - method 1 (not recommended):

          i) retrieve the substation to which this object is connected (for example looking at
             :attr:`GridObjects.line_or_to_subid` [l_id] to know on which substation is connected the origin of
             powerline with id $l_id$.)
          ii) once this substation id is known, compute which are the components of the topological vector that encodes
              information about this substation. For example, if the substation id `sub_id` is 4, we a) count the number
              of elements in substations with id 0, 1, 2 and 3 (say it's 42) we know, by definition that the substation
              4 is encoded in ,:attr:`grid2op.Observation.Observation.topo_vect` starting at component 42 and b) this
              substations has :attr:`GridObjects.sub_info` [sub_id] elements (for the sake of the example say it's 5)
              then the end of the vector for substation 4 will be 42+5 = 47. Finally, we got the representation of the
              "local topology" of the substation 4 by looking at
              :attr:`grid2op.Observation.Observation.topo_vect` [42:47].
          iii) retrieve which component of this vector of dimension 5 (remember we assumed substation 4 had 5 elements)
               encodes information about the origin end of the line with id `l_id`. This information is given in
               :attr:`GridObjects.line_or_to_sub_pos` [l_id]. This is a number between 0 and 4, say it's 3. 3 being
               the index of the object in the substation)

        - method 2 (not recommended): all of the above is stored (for the same powerline) in the
          :attr:`GridObjects.line_or_pos_topo_vect` [l_id]. In the example above, we will have:
          :attr:`GridObjects.line_or_pos_topo_vect` [l_id] = 45 (=42+3:
          42 being the index on which the substation started and 3 being the index of the object in the substation)
        - method 3 (recommended): use any of the function that computes it for you:
          :func:`grid2op.Observation.Observation.state_of` is such an interesting method. The two previous methods
          "method 1" and "method 2" were presented as a way to give detailed and "concrete" example on how the
          modeling of the powergrid work.



    For a given powergrid, this object should be initialized once in the :class:`grid2op.Backend.Backend` when
    the first call to :func:`grid2op.Backend.Backend.load_grid` is performed. In particular the following attributes
    must necessarily be defined (see above for a detailed description of some of the attributes):

    - :attr:`GridObjects.n_line`
    - :attr:`GridObjects.n_gen`
    - :attr:`GridObjects.n_load`
    - :attr:`GridObjects.n_sub`
    - :attr:`GridObjects.sub_info`
    - :attr:`GridObjects.dim_topo`
    - :attr:`GridObjects.load_to_subid`
    - :attr:`GridObjects.gen_to_subid`
    - :attr:`GridObjects.line_or_to_subid`
    - :attr:`GridObjects.line_ex_to_subid`
    - :attr:`GridObjects.load_to_sub_pos`
    - :attr:`GridObjects.gen_to_sub_pos`
    - :attr:`GridObjects.line_or_to_sub_pos`
    - :attr:`GridObjects.line_ex_to_sub_pos`

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

    These information are loaded using the :func:`grid2op.Backend.Backend.load_redispacthing_data` method.

    A call to the function :func:`GridObjects._compute_pos_big_topo` allow to compute the \*_pos_topo_vect attributes
    (for example :attr:`GridObjects.line_ex_pos_topo_vect`) can be computed from this data.

    **NB** it does not store any information about the current state of the powergrid. It stores information that
    cannot be modified by the Agent, the Environment or any other entity.

    Attributes
    ----------

    n_line: :class:`int`
        number of powerlines in the powergrid

    n_gen: :class:`int`
        number of generators in the powergrid

    n_load: :class:`int`
        number of loads in the

    n_sub: :class:`int`
        number of loads in the powergrid

    dim_topo: :class:`int`
        The total number of objects in the powergrid. This is also the dimension of the "topology vector" defined above.

    sub_info: :class:`numpy.ndarray`, dtype:int
        for each substation, gives the number of elements connected to it

    load_to_subid: :class:`numpy.ndarray`, dtype:int
        for each load, gives the id the substation to which it is connected. For example,
        :attr:`GridObjects.load_to_subid` [load_id] gives the id of the substation to which the load of id
        `load_id` is connected.

    gen_to_subid: :class:`numpy.ndarray`, dtype:int
        for each generator, gives the id the substation to which it is connected

    line_or_to_subid: :class:`numpy.ndarray`, dtype:int
        for each line, gives the id the substation to which its "origin" end is connected

    line_ex_to_subid: :class:`numpy.ndarray`, dtype:int
        for each line, gives the id the substation to which its "extremity" end is connected

    load_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        Suppose you represent the topoology of the substation *s* with a vector (each component of this vector will
        represent an object connected to this substation). This vector has, by definition the size
        :attr:`GridObject.sub_info` [s]. `load_to_sub_pos` tells which component of this vector encodes the
        current load. Suppose that load of id `l` is connected to the substation of id `s` (this information is
        stored in :attr:`GridObjects.load_to_subid` [l]), then if you represent the topology of the substation
        `s` with a vector `sub_topo_vect`, then "`sub_topo_vect` [ :attr:`GridObjects.load_to_subid` [l] ]" will encode
        on which bus the load of id `l` is stored.

    gen_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for generators.

    line_or_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos`  but for "origin" end of powerlines.

    line_ex_to_sub_pos: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for "extremity" end of powerlines.

    load_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        The topology if the entire grid is given by a vector, say *topo_vect* of size
        :attr:`GridObjects.dim_topo`. For a given load of id *l*,
        :attr:`GridObjects.load_to_sub_pos` [l] is the index
        of the load *l* in the vector :attr:`grid2op.Observation.Observation.topo_vect` . This means that, if
        "`topo_vect` [ :attr:`GridObjects.load_pos_topo_vect` \[l\] ]=2"
        then load of id *l* is connected to the second bus of the substation.

    gen_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for generators.

    line_or_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for "origin" end of powerlines.

    line_ex_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for "extremity" end of powerlines.

    name_load: :class:`numpy.ndarray`, dtype:str
        ordered names of the loads in the grid.

    name_gen: :class:`numpy.ndarray`, dtype:str
        ordered names of the productions in the grid.

    name_line: :class:`numpy.ndarray`, dtype:str
        ordered names of the powerline in the grid.

    name_sub: :class:`numpy.ndarray`, dtype:str
        ordered names of the substation in the grid

    attr_list_vect: ``list``
        List of string. It represents the attributes that will be stored to/from vector when the Observation is converted
        to/from it. This parameter is also used to compute automatically :func:`GridObjects.dtype` and
        :func:`GridObjects.shape` as well as :func:`GridObjects.size`. If this class is derived, then it's really
        important that this vector is properly set. All the attributes with the name on this vector should have
        consistently the same size and shape, otherwise, some methods will not behave as expected.

    _vectorized: :class:`numpy.ndarray`, dtype:float
        The representation of the GridObject as a vector. See the help of :func:`GridObjects.to_vect` and
        :func:`GridObjects.from_vect` for more information. **NB** for performance reason, the conversion of the internal
        representation to a vector is not performed at any time. It is only performed when :func:`GridObjects.to_vect` is
        called the first time. Otherwise, this attribute is set to ``None``.

    gen_type: :class:`numpy.ndarray`, dtype:str
        Type of the generators, among: "solar", "wind", "hydro", "thermal" and "nuclear". Optional. Used
        for unit commitment problems or redispacthing action.

    gen_pmin: :class:`numpy.ndarray`, dtype:float
        Minimum active power production needed for a generator to work properly. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_pmax: :class:`numpy.ndarray`, dtype:float
        Maximum active power production needed for a generator to work properly. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_redispatchable: :class:`numpy.ndarray`, dtype:bool
        For each generator, it says if the generator is dispatchable or not. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_max_ramp_up: :class:`numpy.ndarray`, dtype:float
        Maximum active power variation possible between two consecutive timestep for each generator:
        a redispatching action
        on generator `g_id` cannot be above :attr:`GridObjects.gen_ramp_up_max` [`g_id`]. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_max_ramp_down: :class:`numpy.ndarray`, dtype:float
        Minimum active power variationpossible between two consecutive timestep for each generator: a redispatching
        action
        on generator `g_id` cannot be below :attr:`GridObjects.gen_ramp_down_min` [`g_id`]. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_min_uptime: :class:`numpy.ndarray`, dtype:float
        The minimum time (expressed in the number of timesteps) a generator needs to be turned on: it's not possible to
        turn off generator `gen_id` that has been turned on less than `gen_min_time_on` [`gen_id`] timesteps
        ago. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_min_downtime: :class:`numpy.ndarray`, dtype:float
        The minimum time (expressed in the number of timesteps) a generator needs to be turned off: it's not possible to
        turn on generator `gen_id` that has been turned off less than `gen_min_time_on` [`gen_id`] timesteps
        ago. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_cost_per_MW: :class:`numpy.ndarray`, dtype:float
        For each generator, it gives the "operating cost", eg the cost, in terms of "used currency" for the production
        of one MW with this generator, if it is already turned on. It's a positive real number. It's the marginal cost
        for each MW. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_startup_cost: :class:`numpy.ndarray`, dtype:float
        The cost to start a generator. It's a positive real number. Optional. Used
        for unit commitment problems or redispacthing action.

    gen_shutdown_cost: :class:`numpy.ndarray`, dtype:float
        The cost to shut down a generator. It's a positive real number. Optional. Used
        for unit commitment problems or redispacthing action.

    redispatching_unit_commitment_availble: ``bool``
        Does the current grid allow for redispatching and / or unit commit problem. If not, any attempt to use it
        will raise a :class:`grid2op.Exceptions.UnitCommitorRedispachingNotAvailable` error.
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

    """
    def __init__(self):
        # name of the objects
        self.name_load = None
        self.name_gen = None
        self.name_line = None
        self.name_sub = None

        self.n_gen = -1
        self.n_load = -1
        self.n_line = -1
        self.n_sub = -1

        self.sub_info = None
        self.dim_topo = -1

        # to which substation is connected each element
        self.load_to_subid = None
        self.gen_to_subid = None
        self.line_or_to_subid = None
        self.line_ex_to_subid = None

        # which index has this element in the substation vector
        self.load_to_sub_pos = None
        self.gen_to_sub_pos = None
        self.line_or_to_sub_pos = None
        self.line_ex_to_sub_pos = None

        # which index has this element in the topology vector
        self.load_pos_topo_vect = None
        self.gen_pos_topo_vect = None
        self.line_or_pos_topo_vect = None
        self.line_ex_pos_topo_vect = None

        # list of attribute to convert it from/to a vector
        self.attr_list_vect = None
        self._vectorized = None

        # for redispatching / unit commitment
        TODO = "TODO COMPLETE THAT BELLOW!!! AND UPDATE THE init methods"
        self._li_attr_disp = ["gen_type", "gen_pmin", "gen_pmax", "gen_redispatchable", "gen_max_ramp_up",
                              "gen_max_ramp_down", "gen_min_uptime", "gen_min_downtime", "gen_cost_per_MW",
                              "gen_startup_cost", "gen_shutdown_cost"]

        self._type_attr_disp = [str, float, float, bool, float, float, int, int, float, float, float]

        self.gen_type = None
        self.gen_pmin = None
        self.gen_pmax = None
        self.gen_redispatchable = None
        self.gen_max_ramp_up = None
        self.gen_max_ramp_down = None
        self.gen_min_uptime = None
        self.gen_min_downtime = None
        self.gen_cost_per_MW = None  # marginal cost
        self.gen_startup_cost = None  # start cost
        self.gen_shutdown_cost = None  # shutdown cost
        self.redispatching_unit_commitment_availble = False

    def _raise_error_attr_list_none(self):
        """
        Raise a "NotImplementedError" if :attr:`GridObjects.attr_list_vect` is not defined.

        Raises
        -------
        ``NotImplementedError``

        """
        if self.attr_list_vect is None:
            raise NotImplementedError("attr_list_vect attribute is not defined for class {}. "
                                      "It is not possible to convert it from/to a vector, "
                                      "nor to know its size, shape or dtype.".format(type(self)))

    def _get_array_from_attr_name(self, attr_name):
        """
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
        return np.array(self.__dict__[attr_name]).flatten()

    def to_vect(self):
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

        """

        if self._vectorized is None:
            self._raise_error_attr_list_none()
            self._vectorized = np.concatenate([self._get_array_from_attr_name(el).astype(np.float)
                                              for el in self.attr_list_vect])
        return self._vectorized

    def shape(self):
        """
        The shapes of all the components of the action, mainly used for gym compatibility is the shape of all
        part of the action.

        It is a numpy integer array.

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
        """
        self._raise_error_attr_list_none()
        res = np.array([self._get_array_from_attr_name(el).shape[0] for el in self.attr_list_vect])
        return res

    def dtype(self):
        """
        The types of the components of the GridObjects, mainly used for gym compatibility is the shape of all part
        of the action.

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
        """

        self._raise_error_attr_list_none()
        res = np.array([self._get_array_from_attr_name(el).dtype for el in self.attr_list_vect])
        return res

    def _assign_attr_from_name(self, attr_nm, vect):
        """
        Assign the proper attributes with name 'attr_nm' with the value of the vector vect

        If this function is overloaded, then the _get_array_from_attr_name must be too.

        Parameters
        ----------
        attr_nm
        vect

        Returns
        -------
        ``None``
        """
        self.__dict__[attr_nm] = vect

    def check_space_legit(self):
        pass

    def from_vect(self, vect):
        """
        Convert a GridObjects, represented as a vector, into an GridObjects object.

        **NB**: in case the class GridObjects is derived,
        either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
        redefined.

        Only the size is checked. If it does not match, an :class:`grid2op.Exceptions.AmbiguousAction` is thrown.
        Otherwise the component of the vector are coerced into the proper type silently.

        It may results in an non deterministic behaviour if the input vector is not a real action, or cannot be
        converted to one.

        Parameters
        ----------
        vect: ``numpy.ndarray``
            A vector representing an Action.

        """

        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while load a GridObjects "
                                            "from a vector. Found {} elements instead of {}".format(
                vect.shape[0], self.size()))

        if np.any(~np.isfinite(vect)):
            raise AmbiguousAction("The action you provided contained not finite number. It cannot be converted to an"
                                  " action class.")

        self._raise_error_attr_list_none()
        prev_ = 0
        for attr_nm, sh, dt in zip(self.attr_list_vect, self.shape(), self.dtype()):
            self._assign_attr_from_name(attr_nm, vect[prev_:(prev_ + sh)].astype(dt))
            prev_ += sh
        self.check_space_legit()

    def size(self):
        """
        When the action is converted to a vector, this method return its size.

        NB that it is a requirement that converting an GridObjects gives a vector of a fixed size throughout a training.

        **NB**: in case the class GridObjects is derived,
        either :attr:`GridObjects.attr_list_vect` is properly defined for the derived class, or this function must be
        redefined.

        Returns
        -------
        size: ``int``
            The size of the GridObjects if it's converted to a flat vector.

        """
        res = np.sum(self.shape())
        return res

    def init_grid_vect(self, name_prod, name_load, name_line, name_sub, sub_info,
                       load_to_subid, gen_to_subid, line_or_to_subid, line_ex_to_subid,
                       load_to_sub_pos, gen_to_sub_pos, line_or_to_sub_pos, line_ex_to_sub_pos,
                       load_pos_topo_vect, gen_pos_topo_vect, line_or_pos_topo_vect, line_ex_pos_topo_vect):
        """
        Initialize the object from the vectors representing the grid.

        Parameters
        ----------
        name_prod: :class:`numpy.ndarray`, dtype:str
            Used to initialized :attr:`GridObjects.name_gen`

        name_load: :class:`numpy.ndarray`, dtype:str
            Used to initialized :attr:`GridObjects.name_load`

        name_line: :class:`numpy.ndarray`, dtype:str
            Used to initialized :attr:`GridObjects.name_line`

        name_sub: :class:`numpy.ndarray`, dtype:str
            Used to initialized :attr:`GridObjects.name_sub`

        sub_info: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.sub_info`

        load_to_subid: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.load_to_subid`

        gen_to_subid: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.gen_to_subid`

        line_or_to_subid: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.line_or_to_subid`

        line_ex_to_subid: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.line_ex_to_subid`

        load_to_sub_pos: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.load_to_sub_pos`

        gen_to_sub_pos: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.gen_to_sub_pos`

        line_or_to_sub_pos: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.line_or_to_sub_pos`

        line_ex_to_sub_pos: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.line_ex_to_sub_pos`

        load_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.load_pos_topo_vect`

        gen_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.gen_pos_topo_vect`

        line_or_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.line_or_pos_topo_vect`

        line_ex_pos_topo_vect: :class:`numpy.ndarray`, dtype:int
            Used to initialized :attr:`GridObjects.line_ex_pos_topo_vect`
        """

        self.name_gen = name_prod
        self.name_load = name_load
        self.name_line = name_line
        self.name_sub = name_sub

        self.n_gen = len(name_prod)
        self.n_load = len(name_load)
        self.n_line = len(name_line)
        self.n_sub = len(name_sub)

        self.sub_info = sub_info
        self.dim_topo = np.sum(sub_info)

        # to which substation is connected each element
        self.load_to_subid = load_to_subid
        self.gen_to_subid = gen_to_subid
        self.line_or_to_subid = line_or_to_subid
        self.line_ex_to_subid = line_ex_to_subid

        # which index has this element in the substation vector
        self.load_to_sub_pos = load_to_sub_pos
        self.gen_to_sub_pos = gen_to_sub_pos
        self.line_or_to_sub_pos = line_or_to_sub_pos
        self.line_ex_to_sub_pos = line_ex_to_sub_pos

        # which index has this element in the topology vector
        self.load_pos_topo_vect = load_pos_topo_vect
        self.gen_pos_topo_vect = gen_pos_topo_vect
        self.line_or_pos_topo_vect = line_or_pos_topo_vect
        self.line_ex_pos_topo_vect = line_ex_pos_topo_vect

    def _aux_pos_big_topo(self, vect_to_subid, vect_to_sub_pos):
        """
        Return the proper "_pos_big_topo" vector given "to_subid" vector and "to_sub_pos" vectors.
        This function is also called to performed sanity check after the load on the powergrid.

        :param vect_to_subid: vector of int giving the id of the topology for this element
        :type vect_to_subid: iterable int

        :param vect_to_sub_pos: vector of int giving the id IN THE SUBSTATION for this element
        :type vect_to_sub_pos: iterable int

        :return:
        """
        res = np.zeros(shape=vect_to_subid.shape)
        for i, (sub_id, my_pos) in enumerate(zip(vect_to_subid, vect_to_sub_pos)):
            obj_before = np.sum(self.sub_info[:sub_id])
            res[i] = obj_before + my_pos
        return res

    def _compute_pos_big_topo(self):
        """
        Compute the position of each element in the big topological vector.

        Topology action are represented by numpy vector of size np.sum(self.sub_info).
        The vector self.load_pos_topo_vect will give the index of each load in this big topology vector.
        For examaple, for load i, self.load_pos_topo_vect[i] gives the position in such a topology vector that
        affect this load.

        This position can be automatically deduced from self.sub_info, self.load_to_subid and self.load_to_sub_pos.

        This is the same for generators and both end of powerlines

        :return: ``None``
        """
        # self.assert_grid_correct()
        self.load_pos_topo_vect = self._aux_pos_big_topo(self.load_to_subid, self.load_to_sub_pos).astype(np.int)
        self.gen_pos_topo_vect = self._aux_pos_big_topo(self.gen_to_subid, self.gen_to_sub_pos).astype(np.int)
        self.line_or_pos_topo_vect = self._aux_pos_big_topo(self.line_or_to_subid, self.line_or_to_sub_pos).astype(np.int)
        self.line_ex_pos_topo_vect = self._aux_pos_big_topo(self.line_ex_to_subid, self.line_ex_to_sub_pos).astype(np.int)

    def assert_grid_correct(self):
        """
        Performs some checking on the loaded _grid to make sure it is consistent.
        It also makes sure that the vector such as *sub_info*, *load_to_subid* or *gen_to_sub_pos* are of the
        right type eg. numpy.ndarray with dtype: np.int

        It is called after the _grid has been loaded.

        These function is by default called by the :class:`grid2op.Environment` class after the initialization of the
        environment.
        If these tests are not successfull, no guarantee are given that the backend will return consistent computations.

        In order for the backend to fully understand the structure of actions, it is strongly advised NOT to override
        this method.

        :return: ``None``
        :raise: :class:`grid2op.EnvError` and possibly all of its derived class.
        """

        if self.name_line is None:
            raise EnvError("name_line is None. Powergrid is invalid. Line names are used to make the correspondance "
                           "between the chronics and the backend")
        if self.name_load is None:
            raise EnvError("name_load is None. Powergrid is invalid. Line names are used to make the correspondance "
                           "between the chronics and the backend")
        if self.name_gen is None:
            raise EnvError("name_gen is None. Powergrid is invalid. Line names are used to make the correspondance "
                           "between the chronics and the backend")
        if self.name_sub is None:
            raise EnvError("name_sub is None. Powergrid is invalid. Substation names are used to make the "
                           "correspondance between the chronics and the backend")

        if self.n_gen <= 0:
            raise EnvError("n_gen is negative. Powergrid is invalid: there are no generator")
        if self.n_load <= 0:
            raise EnvError("n_load is negative. Powergrid is invalid: there are no load")
        if self.n_line <= 0:
            raise EnvError("n_line is negative. Powergrid is invalid: there are no line")
        if self.n_sub <= 0:
            raise EnvError("n_sub is negative. Powergrid is invalid: there are no substation")

        # test if vector can be properly converted
        if not isinstance(self.sub_info, np.ndarray):
            try:
                self.sub_info = np.array(self.sub_info)
                self.sub_info = self.sub_info.astype(np.int)
            except Exception as e:
                raise EnvError("self.sub_info should be convertible to a numpy array")

        if not isinstance(self.load_to_subid, np.ndarray):
            try:
                self.load_to_subid = np.array(self.load_to_subid)
                self.load_to_subid = self.load_to_subid.astype(np.int)
            except Exception as e:
                raise EnvError("self.load_to_subid should be convertible to a numpy array")
        if not isinstance(self.gen_to_subid, np.ndarray):
            try:
                self.gen_to_subid = np.array(self.gen_to_subid)
                self.gen_to_subid = self.gen_to_subid.astype(np.int)
            except Exception as e:
                raise EnvError("self.gen_to_subid should be convertible to a numpy array")
        if not isinstance(self.line_or_to_subid, np.ndarray):
            try:
                self.line_or_to_subid = np.array(self.line_or_to_subid)
                self.line_or_to_subid = self.line_or_to_subid .astype(np.int)
            except Exception as e:
                raise EnvError("self.line_or_to_subid should be convertible to a numpy array")
        if not isinstance(self.line_ex_to_subid, np.ndarray):
            try:
                self.line_ex_to_subid = np.array(self.line_ex_to_subid)
                self.line_ex_to_subid = self.line_ex_to_subid.astype(np.int)
            except Exception as e:
                raise EnvError("self.line_ex_to_subid should be convertible to a numpy array")

        if not isinstance(self.load_to_sub_pos, np.ndarray):
            try:
                self.load_to_sub_pos = np.array(self.load_to_sub_pos)
                self.load_to_sub_pos = self.load_to_sub_pos.astype(np.int)
            except Exception as e:
                raise EnvError("self.load_to_sub_pos should be convertible to a numpy array")
        if not isinstance(self.gen_to_sub_pos, np.ndarray):
            try:
                self.gen_to_sub_pos = np.array(self.gen_to_sub_pos)
                self.gen_to_sub_pos = self.gen_to_sub_pos.astype(np.int)
            except Exception as e:
                raise EnvError("self.gen_to_sub_pos should be convertible to a numpy array")
        if not isinstance(self.line_or_to_sub_pos, np.ndarray):
            try:
                self.line_or_to_sub_pos = np.array(self.line_or_to_sub_pos)
                self.line_or_to_sub_pos = self.line_or_to_sub_pos.astype(np.int)
            except Exception as e:
                raise EnvError("self.line_or_to_sub_pos should be convertible to a numpy array")
        if not isinstance(self.line_ex_to_sub_pos, np.ndarray):
            try:
                self.line_ex_to_sub_pos = np.array(self.line_ex_to_sub_pos)
                self.line_ex_to_sub_pos = self.line_ex_to_sub_pos .astype(np.int)
            except Exception as e:
                raise EnvError("self.line_ex_to_sub_pos should be convertible to a numpy array")

        if not isinstance(self.load_pos_topo_vect, np.ndarray):
            try:
                self.load_pos_topo_vect = np.array(self.load_pos_topo_vect)
                self.load_pos_topo_vect = self.load_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self.load_pos_topo_vect should be convertible to a numpy array")
        if not isinstance(self.gen_pos_topo_vect, np.ndarray):
            try:
                self.gen_pos_topo_vect = np.array(self.gen_pos_topo_vect)
                self.gen_pos_topo_vect = self.gen_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self.gen_pos_topo_vect should be convertible to a numpy array")
        if not isinstance(self.line_or_pos_topo_vect, np.ndarray):
            try:
                self.line_or_pos_topo_vect = np.array(self.line_or_pos_topo_vect)
                self.line_or_pos_topo_vect = self.line_or_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self.line_or_pos_topo_vect should be convertible to a numpy array")
        if not isinstance(self.line_ex_pos_topo_vect, np.ndarray):
            try:
                self.line_ex_pos_topo_vect = np.array(self.line_ex_pos_topo_vect)
                self.line_ex_pos_topo_vect = self.line_ex_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self.line_ex_pos_topo_vect should be convertible to a numpy array")

        # test that all numbers are finite:
        tmp = np.concatenate((
            self.sub_info.flatten(),
                             self.load_to_subid.flatten(),
                             self.gen_to_subid.flatten(),
                             self.line_or_to_subid.flatten(),
                             self.line_ex_to_subid.flatten(),
                             self.load_to_sub_pos.flatten(),
                             self.gen_to_sub_pos.flatten(),
                             self.line_or_to_sub_pos.flatten(),
                             self.line_ex_to_sub_pos.flatten(),
                             self.load_pos_topo_vect.flatten(),
                             self.gen_pos_topo_vect.flatten(),
                             self.line_or_pos_topo_vect.flatten(),
                             self.line_ex_pos_topo_vect.flatten()
                              ))
        try:
            if np.any(~np.isfinite(tmp)):
                raise EnvError("One of the vector is made of non finite elements")
        except Exception as e:
            raise EnvError("Impossible to check wheter or not vectors contains online finite elements (pobably one "
                           "or more topology related vector is not valid (None)")

        # check sizes
        if len(self.sub_info) != self.n_sub:
            raise IncorrectNumberOfSubstation("The number of substation is not consistent in "
                                              "self.sub_info (size \"{}\")"
                                              " and  self.n_sub ({})".format(len(self.sub_info), self.n_sub))
        if np.sum(self.sub_info) != self.n_load + self.n_gen + 2*self.n_line:
            err_msg = "The number of elements of elements is not consistent between self.sub_info where there are "
            err_msg +=  "{} elements connected to all substations and the number of load, generators and lines in " \
                        "the _grid."
            err_msg = err_msg.format(np.sum(self.sub_info))
            raise IncorrectNumberOfElements(err_msg)

        if len(self.load_to_subid) != self.n_load:
            raise IncorrectNumberOfLoads()
        if len(self.gen_to_subid) != self.n_gen:
            raise IncorrectNumberOfGenerators()
        if len(self.line_or_to_subid) != self.n_line:
            raise IncorrectNumberOfLines()
        if len(self.line_ex_to_subid) != self.n_line:
            raise IncorrectNumberOfLines()

        if len(self.load_to_sub_pos) != self.n_load:
            raise IncorrectNumberOfLoads()
        if len(self.gen_to_sub_pos) != self.n_gen:
            raise IncorrectNumberOfGenerators()
        if len(self.line_or_to_sub_pos) != self.n_line:
            raise IncorrectNumberOfLines()
        if len(self.line_ex_to_sub_pos) != self.n_line:
            raise IncorrectNumberOfLines()

        if len(self.load_pos_topo_vect) != self.n_load:
            raise IncorrectNumberOfLoads()
        if len(self.gen_pos_topo_vect) != self.n_gen:
            raise IncorrectNumberOfGenerators()
        if len(self.line_or_pos_topo_vect) != self.n_line:
            raise IncorrectNumberOfLines()
        if len(self.line_ex_pos_topo_vect) != self.n_line:
            raise IncorrectNumberOfLines()

        # test if object are connected to right substation
        obj_per_sub = np.zeros(shape=(self.n_sub,))
        for sub_id in self.load_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in self.gen_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in self.line_or_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in self.line_ex_to_subid:
            obj_per_sub[sub_id] += 1

        if not np.all(obj_per_sub == self.sub_info):
            raise IncorrectNumberOfElements()

        # test right number of element in substations
        # test that for each substation i don't have an id above the number of element of a substations
        for i, (sub_id, sub_pos) in enumerate(zip(self.load_to_subid, self.load_to_sub_pos)):
            if sub_pos >= self.sub_info[sub_id]:
                raise IncorrectPositionOfLoads("for load {}".format(i))
        for i, (sub_id, sub_pos) in enumerate(zip(self.gen_to_subid, self.gen_to_sub_pos)):
            if sub_pos >= self.sub_info[sub_id]:
                raise IncorrectPositionOfGenerators("for generator {}".format(i))
        for i, (sub_id, sub_pos) in enumerate(zip(self.line_or_to_subid, self.line_or_to_sub_pos)):
            if sub_pos >= self.sub_info[sub_id]:
                raise IncorrectPositionOfLines("for line {} at origin end".format(i))
        for i, (sub_id, sub_pos) in enumerate(zip(self.line_ex_to_subid, self.line_ex_to_sub_pos)):
            if sub_pos >= self.sub_info[sub_id]:
                # pdb.set_trace()
                raise IncorrectPositionOfLines("for line {} at extremity end".format(i))

        # check that i don't have 2 objects with the same id in the "big topo" vector
        if len(np.unique(np.concatenate((self.load_pos_topo_vect.flatten(),
                                        self.gen_pos_topo_vect.flatten(),
                                        self.line_or_pos_topo_vect.flatten(),
                                        self.line_ex_pos_topo_vect.flatten())))) != np.sum(self.sub_info):
                raise EnvError("2 different objects would have the same id in the topology vector.")

        # check that self.load_pos_topo_vect and co are consistent
        load_pos_big_topo = self._aux_pos_big_topo(self.load_to_subid, self.load_to_sub_pos)
        if not np.all(load_pos_big_topo == self.load_pos_topo_vect):
            raise IncorrectPositionOfLoads()
        gen_pos_big_topo = self._aux_pos_big_topo(self.gen_to_subid, self.gen_to_sub_pos)
        if not np.all(gen_pos_big_topo == self.gen_pos_topo_vect):
            raise IncorrectNumberOfGenerators()
        lines_or_pos_big_topo = self._aux_pos_big_topo(self.line_or_to_subid, self.line_or_to_sub_pos)
        if not np.all(lines_or_pos_big_topo == self.line_or_pos_topo_vect):
            raise IncorrectPositionOfLines()
        lines_ex_pos_big_topo = self._aux_pos_big_topo(self.line_ex_to_subid, self.line_ex_to_sub_pos)
        if not np.all(lines_ex_pos_big_topo == self.line_ex_pos_topo_vect):
            raise IncorrectPositionOfLines()

        # no empty bus: at least one element should be present on each bus
        if np.any(self.sub_info < 1):
            raise BackendError("There are {} bus with 0 element connected to it.".format(np.sum(self.sub_info < 1)))

        # redispatching / unit commitment
        if self.redispatching_unit_commitment_availble:
            if self.gen_type is None:
                raise InvalidRedispatching("Impossible to recognize the type of generators (gen_type) when "
                                           "redispatching is supposed to be available.")
            if self.gen_pmin is None:
                raise InvalidRedispatching("Impossible to recognize the pmin of generators (gen_pmin) when "
                                           "redispatching is supposed to be available.")
            if self.gen_pmax is None:
                raise InvalidRedispatching("Impossible to recognize the pmax of generators (gen_pmax) when "
                                           "redispatching is supposed to be available.")
            if self.gen_redispatchable is None:
                raise InvalidRedispatching("Impossible to know which generator can be dispatched (gen_redispatchable)"
                                           " when redispatching is supposed to be available.")
            if self.gen_max_ramp_up is None:
                raise InvalidRedispatching("Impossible to recognize the ramp up of generators (gen_max_ramp_up)"
                                           " when redispatching is supposed to be available.")
            if self.gen_max_ramp_down is None:
                raise InvalidRedispatching("Impossible to recognize the ramp up of generators (gen_max_ramp_down)"
                                           " when redispatching is supposed to be available.")
            if self.gen_min_uptime is None:
                raise InvalidRedispatching("Impossible to recognize the min uptime of generators (gen_min_uptime)"
                                           " when redispatching is supposed to be available.")
            if self.gen_min_downtime is None:
                raise InvalidRedispatching("Impossible to recognize the min downtime of generators (gen_min_downtime)"
                                           " when redispatching is supposed to be available.")
            if self.gen_cost_per_MW is None:
                raise InvalidRedispatching("Impossible to recognize the marginal costs of generators (gen_cost_per_MW)"
                                           " when redispatching is supposed to be available.")
            if self.gen_startup_cost is None:
                raise InvalidRedispatching("Impossible to recognize the start up cost of generators (gen_startup_cost)"
                                           " when redispatching is supposed to be available.")
            if self.gen_shutdown_cost is None:
                raise InvalidRedispatching("Impossible to recognize the shut down cost of generators "
                                           "(gen_shutdown_cost) when redispatching is supposed to be available.")

            if len(self.gen_type) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the type of generators (gen_type) when "
                                           "redispatching is supposed to be available.")
            if len(self.gen_pmin) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the pmin of generators (gen_pmin) when "
                                           "redispatching is supposed to be available.")
            if len(self.gen_pmax) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the pmax of generators (gen_pmax) when "
                                           "redispatching is supposed to be available.")
            if len(self.gen_redispatchable) != self.n_gen:
                raise InvalidRedispatching("Invalid length for which generator can be dispatched (gen_redispatchable)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_max_ramp_up) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the ramp up of generators (gen_max_ramp_up)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_max_ramp_down) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the ramp up of generators (gen_max_ramp_down)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_min_uptime) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the min uptime of generators (gen_min_uptime)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_min_downtime) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the min downtime of generators (gen_min_downtime)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_cost_per_MW) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the marginal costs of generators (gen_cost_per_MW)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_startup_cost) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the start up cost of generators (gen_startup_cost)"
                                           " when redispatching is supposed to be available.")
            if len(self.gen_shutdown_cost) != self.n_gen:
                raise InvalidRedispatching("Invalid length for the shut down cost of generators "
                                           "(gen_shutdown_cost) when redispatching is supposed to be available.")

            if np.any(self.gen_min_uptime < 0):
                raise InvalidRedispatching("Minimum uptime of generator (gen_min_uptime) cannot be negative")
            if np.any(self.gen_min_downtime < 0):
                raise InvalidRedispatching("Minimum downtime of generator (gen_min_downtime) cannot be negative")
            
            for el in self.gen_type:
                if not el in ["solar", "wind", "hydro", "thermal", "nuclear"]:
                    raise InvalidRedispatching("Unknown generator type : {}".format(el))

            if np.any(self.gen_pmin < 0.):
                raise InvalidRedispatching("One of the Pmin (gen_pmin) is negative")
            if np.any(self.gen_pmax < 0.):
                raise InvalidRedispatching("One of the Pmax (gen_pmax) is negative")
            if np.any(self.gen_max_ramp_down < 0.):
                raise InvalidRedispatching("One of the ramp up (gen_max_ramp_down) is negative")
            if np.any(self.gen_max_ramp_up < 0.):
                raise InvalidRedispatching("One of the ramp down (gen_max_ramp_up) is negative")
            if np.any(self.gen_startup_cost < 0.):
                raise InvalidRedispatching("One of the start up cost (gen_startup_cost) is negative")
            if np.any(self.gen_shutdown_cost < 0.):
                raise InvalidRedispatching("One of the start up cost (gen_shutdown_cost) is negative")

            for el, type_ in zip(["gen_type", "gen_pmin", "gen_pmax", "gen_redispatchable", "gen_max_ramp_up",
                                  "gen_max_ramp_down", "gen_min_uptime", "gen_min_downtime", "gen_cost_per_MW",
                                  "gen_startup_cost", "gen_shutdown_cost"],
                                 [str, np.float, np.float, np.bool, np.float,
                                 np.float, np.int, np.int, np.float,
                                 np.float, np.float]):
                if not isinstance(self.__dict__[el], np.ndarray):
                    try:
                        self.__dict__[el] = np.array(self.__dict__[el])
                    except Exception as e:
                        raise InvalidRedispatching("{} should be convertible to a numpy array".format(el))
                if not np.issubdtype(self.__dict__[el].dtype, np.dtype(type_).type):
                    try:
                        self.__dict__[el] = self.__dict__[el].astype(type_)
                    except Exception as e:
                        raise InvalidRedispatching("{} should be convertible data should be convertible to "
                                                   "{}".format(el, type_))
            if np.any(self.gen_max_ramp_up[self.gen_redispatchable] > self.gen_pmax[self.gen_redispatchable]):
                raise InvalidRedispatching("Invalid maximum ramp for some generator (above pmax)")

    def init_grid(self, gridobj):
        """
        Initialize this :class:`GridObjects` instance with a provided instance.

        It does not perform any check on the validity of the `gridobj` parameters, but it guarantees that  if `gridobj`
        is a valid grid, then the initialization will lead to a valid grid too.

        Parameters
        ----------
        gridobj: :class:`GridObjects`
            The representation of the powergrid
        """
        self.name_gen = gridobj.name_gen
        self.name_load = gridobj.name_load
        self.name_line = gridobj.name_line
        self.name_sub = gridobj.name_sub

        self.n_gen = len(gridobj.name_gen)
        self.n_load = len(gridobj.name_load)
        self.n_line = len(gridobj.name_line)
        self.n_sub = len(gridobj.name_sub)

        self.sub_info = gridobj.sub_info
        self.dim_topo = np.sum(gridobj.sub_info)

        # to which substation is connected each element
        self.load_to_subid = gridobj.load_to_subid
        self.gen_to_subid = gridobj.gen_to_subid
        self.line_or_to_subid = gridobj.line_or_to_subid
        self.line_ex_to_subid = gridobj.line_ex_to_subid

        # which index has this element in the substation vector
        self.load_to_sub_pos = gridobj.load_to_sub_pos
        self.gen_to_sub_pos = gridobj.gen_to_sub_pos
        self.line_or_to_sub_pos = gridobj.line_or_to_sub_pos
        self.line_ex_to_sub_pos = gridobj.line_ex_to_sub_pos

        # which index has this element in the topology vector
        self.load_pos_topo_vect = gridobj.load_pos_topo_vect
        self.gen_pos_topo_vect = gridobj.gen_pos_topo_vect
        self.line_or_pos_topo_vect = gridobj.line_or_pos_topo_vect
        self.line_ex_pos_topo_vect = gridobj.line_ex_pos_topo_vect

        # for redispatching / unit commitment
        self.gen_type = gridobj.gen_type
        self.gen_pmin = gridobj.gen_pmin
        self.gen_pmax = gridobj.gen_pmax
        self.gen_redispatchable = gridobj.gen_redispatchable
        self.gen_max_ramp_up = gridobj.gen_max_ramp_up
        self.gen_max_ramp_down = gridobj.gen_max_ramp_down
        self.gen_min_uptime = gridobj.gen_min_uptime
        self.gen_min_downtime = gridobj.gen_min_downtime
        self.gen_cost_per_MW = gridobj.gen_cost_per_MW
        self.gen_startup_cost = gridobj.gen_startup_cost
        self.gen_shutdown_cost = gridobj.gen_shutdown_cost
        self.redispatching_unit_commitment_availble = gridobj.redispatching_unit_commitment_availble

    def get_obj_connect_to(self, _sentinel=None, substation_id=None):
        """
        Get all the object connected to a given substation:

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        substation_id: ``int``
            ID of the substation we want to inspect

        Returns
        -------
        res: ``dict``
            A dictionnary with keys:

              - "loads_id": a vector giving the id of the loads connected to this substation, empty if none
              - "generators_id": a vector giving the id of the generators connected to this substation, empty if none
              - "lines_or_id": a vector giving the id of the origin end of the powerlines connected to this substation,
                empty if none
              - "lines_ex_id": a vector giving the id of the extermity end of the powerlines connected to this
                substation, empty if none.
              - "nb_elements" : number of elements connected to this substation

        """
        if _sentinel is not None:
            raise Grid2OpException("get_obj_connect_to shoud be used only with key-word arguments")

        if substation_id is None:
            raise Grid2OpException("You ask the composition of a substation without specifying its id."
                                   "Please provide \"substation_id\"")
        if substation_id >= len(self.sub_info):
            raise Grid2OpException("There are no substation of id \"substation_id={}\" in this grid.".format(substation_id))

        res = {}
        res["loads_id"] = np.where(self.load_to_subid == substation_id)[0]
        res["generators_id"] = np.where(self.gen_to_subid == substation_id)[0]
        res["lines_or_id"] = np.where(self.line_or_to_subid == substation_id)[0]
        res["lines_ex_id"] = np.where(self.line_ex_to_subid == substation_id)[0]
        res["nb_elements"] = self.sub_info[substation_id]
        return res

    def get_lines_id(self, _sentinel=None, from_=None, to_=None):
        """
        Returns the list of all the powerlines id in the backend going from `from_` to `to_`

        Parameters
        ----------
        _sentinel: ``None``
            Internal, do not use

        from_: ``int``
            Id the substation to which the origin end of the powerline to look for should be connected to

        to_: ``int``
            Id the substation to which the extremity end of the powerline to look for should be connected to

        Returns
        -------
        res: ``list``
            Id of the powerline looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match is found.

        """
        res = []
        if from_ is None:
            raise BackendError("ObservationSpace.get_lines_id: impossible to look for a powerline with no origin "
                               "substation. Please modify \"from_\" parameter")
        if to_ is None:
            raise BackendError("ObservationSpace.get_lines_id: impossible to look for a powerline with no extremity "
                               "substation. Please modify \"to_\" parameter")

        for i, (ori, ext) in enumerate(zip(self.line_or_to_subid, self.line_ex_to_subid)):
            if ori == from_ and ext == to_:
                res.append(i)

        if res is []:
            raise BackendError("ObservationSpace.get_line_id: impossible to find a powerline with connected at "
                               "origin at {} and extremity at {}".format(from_, to_))

        return res

    def get_generators_id(self, sub_id):
        """
        Returns the list of all generators id in the backend connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the generator

        Returns
        -------
        res: ``list``
            Id of the generator id looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match is found.


        """
        res = []
        if sub_id is None:
            raise BackendError(
                "GridObjects.get_generators_id: impossible to look for a generator not connected to any substation. "
                "Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.gen_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "GridObjects.get_generators_id: impossible to find a generator connected at "
                "substation {}".format(sub_id))

        return res

    def get_loads_id(self, sub_id):
        """
        Returns the list of all generators id in the backend connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the generator

        Returns
        -------
        res: ``list``
            Id of the generator id looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match found.

        """
        res = []
        if sub_id is None:
            raise BackendError(
                "GridObjects.get_loads_id: impossible to look for a load not connected to any substation. "
                "Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.load_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "GridObjects.get_loads_id: impossible to find a load connected at substation {}".format(sub_id))

        return res

    def to_dict(self):
        """
        Convert the object as a dictionnary.
        Note that unless this method is overidden, a call to it will only output the

        Returns
        -------
        res: ``dict``
            The representation of the object as a dictionary that can be json serializable.
        """
        res = {}
        save_to_dict(res, self, "name_gen", lambda li: [str(el) for el in li])
        save_to_dict(res, self, "name_load", lambda li: [str(el) for el in li])
        save_to_dict(res, self, "name_line", lambda li: [str(el) for el in li])
        save_to_dict(res, self, "name_sub", lambda li: [str(el) for el in li])

        save_to_dict(res, self, "sub_info", lambda li: [int(el) for el in li])

        save_to_dict(res, self, "load_to_subid", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "gen_to_subid", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "line_or_to_subid", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "line_ex_to_subid", lambda li: [int(el) for el in li])

        save_to_dict(res, self, "load_to_sub_pos", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "gen_to_sub_pos", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "line_or_to_sub_pos", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "line_ex_to_sub_pos", lambda li: [int(el) for el in li])

        save_to_dict(res, self, "load_pos_topo_vect", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "gen_pos_topo_vect", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "line_or_pos_topo_vect", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "line_ex_pos_topo_vect", lambda li: [int(el) for el in li])

        # redispatching
        if self.redispatching_unit_commitment_availble:
            for nm_attr, type_attr in zip(self._li_attr_disp, self._type_attr_disp):
                save_to_dict(res, self, nm_attr, lambda li: [type_attr(el) for el in li])
        else:
            for nm_attr in self._li_attr_disp:
                res[nm_attr] = None
        return res

    @staticmethod
    def from_dict(dict_):
        """
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
        res = GridObjects()
        res.name_gen = extract_from_dict(dict_, "name_gen", lambda x: np.array(x).astype(str))
        res.name_load = extract_from_dict(dict_, "name_load", lambda x: np.array(x).astype(str))
        res.name_line = extract_from_dict(dict_, "name_line", lambda x: np.array(x).astype(str))
        res.name_sub = extract_from_dict(dict_, "name_sub", lambda x: np.array(x).astype(str))

        res.sub_info = extract_from_dict(dict_, "sub_info", lambda x: np.array(x).astype(np.int))
        res.load_to_subid = extract_from_dict(dict_, "load_to_subid", lambda x: np.array(x).astype(np.int))
        res.gen_to_subid = extract_from_dict(dict_, "gen_to_subid", lambda x: np.array(x).astype(np.int))
        res.line_or_to_subid = extract_from_dict(dict_, "line_or_to_subid", lambda x: np.array(x).astype(np.int))
        res.line_ex_to_subid = extract_from_dict(dict_, "line_ex_to_subid", lambda x: np.array(x).astype(np.int))

        res.load_to_sub_pos = extract_from_dict(dict_, "load_to_sub_pos", lambda x: np.array(x).astype(np.int))
        res.gen_to_sub_pos = extract_from_dict(dict_, "gen_to_sub_pos", lambda x: np.array(x).astype(np.int))
        res.line_or_to_sub_pos = extract_from_dict(dict_, "line_or_to_sub_pos", lambda x: np.array(x).astype(np.int))
        res.line_ex_to_sub_pos = extract_from_dict(dict_, "line_ex_to_sub_pos", lambda x: np.array(x).astype(np.int))

        res.load_pos_topo_vect = extract_from_dict(dict_, "load_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        res.gen_pos_topo_vect = extract_from_dict(dict_, "gen_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        res.line_or_pos_topo_vect = extract_from_dict(dict_, "line_or_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        res.line_ex_pos_topo_vect = extract_from_dict(dict_, "line_ex_pos_topo_vect", lambda x: np.array(x).astype(np.int))

        res.n_gen = len(res.name_gen)
        res.n_load = len(res.name_load)
        res.n_line = len(res.name_line)
        res.n_sub = len(res.name_sub)
        res.dim_topo = np.sum(res.sub_info)

        if dict_["gen_type"] is None:
            res.redispatching_unit_commitment_availble = False
            # and no need to make anything else, because everything is already initialized at None
        else:
            res.redispatching_unit_commitment_availble = True
            type_attr_disp = [str, np.float, np.float, np.bool, np.float, np.float,
                              np.int, np.int, np.float, np.float, np.float]
            for nm_attr, type_attr in zip(res._li_attr_disp, type_attr_disp):
                res.__dict__[nm_attr] = extract_from_dict(dict_, nm_attr, lambda x: np.array(x).astype(type_attr))
        return res


class RandomObject(object):
    """
    Utility class to deal with randomness in some aspect of the game (chronics, action_space, observation_space for
    examples.

    Attributes
    ----------
    space_prng: ``numpy.random.RandomState``
        The random state of the observation (in case of non deterministic observations or Action.
        This should not be used at the
        moment)

    seed_used: ``int``
        The seed used throughout the episode in case of non deterministic observations or action.

    """
    def __init__(self):
        self.space_prng = np.random.RandomState()
        self.seed_used = None

    def seed(self, seed):
        """
        Use to set the seed in case of non deterministic observations.
        :param seed:
        :return:
        """
        self.seed_used = seed
        if self.seed_used is not None:
            # in this case i have specific seed set. So i force the seed to be deterministic.
            self.space_prng.seed(seed=self.seed_used)


class SerializableSpace(GridObjects, RandomObject):
    """
    This class allows to serialize / de serialize the action space or observation space.

    It should not be used inside an Environment, as some functions of the action might not be compatible with
    the serialization, especially the checking of whether or not an Action is legal or not.

    Attributes
    ----------

    subtype: ``type``
        Type use to build the template object :attr:`SerializableSpace.template_obj`. This type should derive
        from :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`.

    _template_obj: :class:`grid2op.GridObjects`
        An instance of the "*subtype*" provided used to provide higher level utilities, such as the size of the
        action (see :func:`grid2op.Action.Action.size`) or to sample a new Action
        (see :func:`grid2op.Action.Action.sample`) for example.

    n: ``int``
        Size of the space

    shape: ``numpy.ndarray``, dtype:int
        Shape of each of the component of the Object if represented in a flat vector. An instance that derives from a
        GridObject (for example :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`) can be
        thought of as being concatenation of independant spaces. This vector gives the dimension of all the basic
        spaces they are made of.

    dtype: ``numpy.ndarray``, dtype:int
        Data type of each of the component of the Object if represented in a flat vector. An instance that derives from
        a GridObject (for example :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`) can be
        thought of as being concatenation of independant spaces. This vector gives the type of all the basic
        spaces they are made of.

    """
    def __init__(self, gridobj,
                 subtype=object):
        """

        subtype: ``type``
            Type of action used to build :attr:`SerializableActionSpace._template_act`. This type should derive
            from :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation` .

        """

        if not isinstance(subtype, type):
            raise Grid2OpException(
                "Parameter \"subtype\" used to build the Space should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(subtype)))

        GridObjects.__init__(self)
        RandomObject.__init__(self)

        self.init_grid(gridobj)

        self.subtype = subtype
        self._template_obj = self.subtype(gridobj=self)
        self.n = self._template_obj.size()

        self.global_vars = None

        self.shape = self._template_obj.shape()
        self.dtype = self._template_obj.dtype()

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionnary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an Observation Space (aka :class:`grid2op.Observation.ObservartionHelper`)
            or the Action Space (aka :class:`grid2op.Action.HelperAction`)
            as a dictionnary.

        Returns
        -------
        res: :class:`SerializableSpace`
            An instance of an SerializableSpace matching the dictionnary.

        """

        if isinstance(dict_, str):
            path = dict_
            if not os.path.exists(path):
                raise Grid2OpException("Unable to find the file \"{}\" to load the ObservationSpace".format(path))
            with open(path, "r", encoding="utf-8") as f:
                dict_ = json.load(fp=f)

        gridobj = GridObjects.from_dict(dict_)

        actionClass_str = extract_from_dict(dict_, "subtype", str)
        actionClass_li = actionClass_str.split('.')

        # pdb.set_trace()
        if actionClass_li[-1] in globals():
            subtype = globals()[actionClass_li[-1]]
        else:
            # TODO make something better and recursive here
            exec("from {} import {}".format(".".join(actionClass_li[:-1]), actionClass_li[-1]))
            try:
                subtype = eval(actionClass_li[-1])
            except NameError:
                if len(actionClass_li) > 1:
                    try:
                        subtype = eval(".".join(actionClass_li[1:]))
                    except:
                        msg_err_ = "Impossible to find the module \"{}\" to load back the space (ERROR 1). " \
                                   "Try \"from {} import {}\""
                        raise Grid2OpException(msg_err_.format(actionClass_str, ".".join(actionClass_li[:-1]),
                                                               actionClass_li[-1]))
                else:
                    msg_err_ = "Impossible to find the module \"{}\" to load back the space (ERROR 2). " \
                               "Try \"from {} import {}\""
                    raise Grid2OpException(msg_err_.format(actionClass_str, ".".join(actionClass_li[:-1]),
                                                           actionClass_li[-1]))
            except AttributeError:
                try:
                    subtype = eval(actionClass_li[-1])
                except:
                    if len(actionClass_li) > 1:
                        msg_err_ = "Impossible to find the class named \"{}\" to load back the space (ERROR 3)" \
                                   "(module is found but not the class in it) Please import it via " \
                                   "\"from {} import {}\"."
                        msg_err_ = msg_err_.format(actionClass_str,
                                                   ".".join(actionClass_li[:-1]),
                                                   actionClass_li[-1])
                    else:
                        msg_err_ = "Impossible to import the class named \"{}\" to load back the space (ERROR 4) " \
                                   "(the module is found but not the class in it)"
                        msg_err_ = msg_err_.format(actionClass_str)
                    raise Grid2OpException(msg_err_)

        res = SerializableSpace(gridobj=gridobj,
                                subtype=subtype)
        return res

    def to_dict(self):
        """
        Serialize this object as a dictionnary.

        Returns
        -------
        res: ``dict``
            A dictionnary representing this object content. It can be loaded back with
             :func:`SerializableObservationSpace.from_dict`
        """
        res = super().to_dict()

        save_to_dict(res, self, "subtype", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))

        return res

    def size(self):
        """
        The size of any action converted to vector.

        Returns
        -------
        n: ``int``
            The size of the action space.
        """
        return self.n

    def from_vect(self, obj_as_vect):
        """
        Convert an action, represented as a vector to a valid :class:`Action` instance

        Parameters
        ----------
        obj_as_vect: ``numpy.ndarray``
            A object living in a space represented as a vector (typically an :class:`grid2op.Action.Action` or an
            :class:`grid2op.Observation.Observation` represented as a numpy vector)

        Returns
        -------
        res: :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`
            The corresponding action (or observation) as an object (and not as a vector). The return type is given
            by the type of :attr:`SerializableSpace._template_obj`

        """
        res = copy.deepcopy(self._template_obj)
        res.from_vect(obj_as_vect)
        return res
