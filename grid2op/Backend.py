"""
This Module defines the template of a backend class.
Backend instances are responsible to translate action (performed either by an Agent or by the Environment) into
comprehensive powergrid modifications.
They are responsible to perform the powerflow (AC or DC) computation.

It is also through the backend that some quantities about the powergrid (such as the flows) can be inspected.

A backend is mandatory for a Grid2Op environment to work.

To be a valid backend, some properties are mandatory:

    - order of objects matters and should be deterministic (for example :func:`Backend.get_line_status`
      shall return the status of the lines always in the same order)
    - order of objects should be the same if the same underlying object is queried (for example, is
      :func:`Backend.get_line_status`\[i\] is the status of the powerline "*toto*", then
      :func:`Backend.get_thermal_limit`\[i\] returns the thermal limits of this same powerline "*toto*")
    - it allows to compute AC and DC powerflow
    - it allows to:

        - change the value consumed (both active and reactive) by each load of the network
        - change the amount of power produced and the voltage setpoint of each generator unit of the powergrid
        - allow for powerline connection / disconnection
        - allow for the modification of the connectivity of the powergrid (change in topology)
        - allow for deep copy.

The order of the values returned are always the same and determined when the backend is loaded by its attribute
'\*_names'. For example, when the ith element of the results of a call to :func:`Backend.get_line_flow` is the
flow on the powerline with name `lines_names[i]`.

"""

import copy

from abc import ABC, abstractmethod
import numpy as np
import warnings

try:
    from .Exceptions import *
except ImportError:
    from Exceptions import *

import pdb


# TODO code a method to give information about element (given name, gives type, substation, bus connected etc.)
# TODO given a bus, returns the names of the elements connected to it
# TODO given a substation, returns the name of the elements connected to it
# TODO given to substations, returns the name of the powerlines connecting them, if any

# TODO URGENT: if chronics are "loop through" multiple times, only last results are saved. :-/


class Backend(ABC):
    """
    This is a base class for each :class:`Backend` object.
    It allows to run power flow smoothly, and abstract the method of computing cascading failures.
    This class allow the user or the agent to interact with an power flow calculator, while relying on dedicated methods to change the power _grid behaviour.

    Attributes
    ----------
    detailed_infos_for_cascading_failures: :class:`bool`
        Whether to be verbose when computing a cascading failure.

    n_lines: :class:`int`
        number of powerline in the _grid

    n_generators: :class:`int`
        number of generators in the _grid

    n_loads: :class:`int`
        number of loads in the powergrid

    n_substations: :class:`int`
        number of substation in the powergrid

    subs_elements: :class:`numpy.array`, dtype:int
        for each substation, gives the number of elements connected to it

    load_to_subid: :class:`numpy.array`, dtype:int
        for each load, gives the id the substation to which it is connected

    gen_to_subid: :class:`numpy.array`, dtype:int
        for each generator, gives the id the substation to which it is connected

    lines_or_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "origin" end is connected

    lines_ex_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "extremity" end is connected

    load_to_sub_pos: :class:`numpy.array`, dtype:int
        The topology if of the subsation *i* is given by a vector, say *sub_topo_vect* of size
        :attr:`Backend.subs_info`\[i\]. For a given load of id *l*, :attr:`Backend._load_to_sub_pos`\[l\] is the index
        of the load *l* in the vector *sub_topo_vect*. This means that, if
        *sub_topo_vect\[ action._load_to_sub_pos\[l\] \]=2*
        then load of id *l* is connected to the second bus of the substation.

    gen_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_to_sub_pos` but for generators.

    lines_or_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_to_sub_pos`  but for "origin" end of powerlines.

    lines_ex_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_to_sub_pos` but for "extremity" end of powerlines.

    load_pos_topo_vect: :class:`numpy.array`, dtype:int
        It has a similar role as :attr:`Backend._load_to_sub_pos` but it gives the position in the vector representing
        the whole topology. More concretely, if the complete topology of the powergrid is represented here by a vector
        *full_topo_vect* resulting of the concatenation of the topology vector for each substation
        (see :attr:`Backend._load_to_sub_pos`for more information). For a load of id *l* in the powergrid,
        :attr:`Backend._load_pos_topo_vect`\[l\] gives the index, in this *full_topo_vect* that concerns load *l*.
        More formally, if *_topo_vect\[ backend._load_pos_topo_vect\[l\] \]=2* then load of id l is connected to the
        second bus of the substation.

    gen_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_pos_topo_vect` but for generators.

    lines_or_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_pos_topo_vect` but for "origin" end of powerlines.

    lines_ex_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_pos_topo_vect` but for "extremity" end of powerlines.

    _grid: (its type depends on the backend, precisely)
        is a representation of the powergrid that can be called and manipulated by the backend.

    name_loads: :class:`numpy.array`, dtype:str
        ordered name of the loads in the backend. This is mainly use to make sure the "chronics" are used properly.

    name_prods: :class:`numpy.array`, dtype:str
        ordered name of the productions in the backend. This is mainly use to make sure the "chronics" are used properly.

    name_lines: :class:`numpy.array`, dtype:str
        ordered name of the productions in the backend. This is mainly use to make sure the "chronics" are used properly.

    name_subs: :class:`numpy.array`, dtype:str
        ordered name of the substation in the _grid. This is mainly use to make sure the "chronics" are used properly.

    thermal_limit_a: :class:`numpy.array`, dtype:float
        Thermal limit of the powerline in amps for each powerline. Thie thermal limit is relevant on only one
        side of the powerline: the same side returned by :func:`Backend.get_line_overflow`
    """
    def __init__(self, detailed_infos_for_cascading_failures=True):
        """
        Initialize an instance of Backend. This does nothing per se. Only the call to :func:`Backend.load_grid`
        should guarantee the backend is properly configured.

        :param detailed_infos_for_cascading_failures: Whether to be detailed (but slow) when computing cascading failures
        :type detailed_infos_for_cascading_failures: :class:`bool`

        """

        # the following parameter is used to control the amount of verbosity when computing a cascading failure
        # if it's set to true, it returns all intermediate _grid states. This can slow down the computation!
        self.detailed_infos_for_cascading_failures = detailed_infos_for_cascading_failures

        self.n_lines = None  # int: number of powerlines
        self.n_generators = None  # int: number of generators
        self.n_loads = None  # int: number of loads
        self.n_substations = None  # int: number of substations
        self.subs_elements = None  # vector[int]: of size number of substation. Tells for each substation the number of element connected to it

        self.load_to_subid = None  # vector[int]: as size number of load, giving for each the substation id to which it is connected
        self.gen_to_subid = None  # vector[int]: as size number of generators, giving for each the substation id to which it is connected
        self.lines_or_to_subid = None  # vector[int]: as size number of lines, giving for each the substation id to which its "origin" end is connected
        self.lines_ex_to_subid = None  # vector[int]: as size number of lines, giving for each the substation id to which its "extremity" end is connected

        # position in the vector of substation
        self.load_to_sub_pos = None   # vector[int]: as size number of load, giving for each the postition of this load in among the element of this substation
        self.gen_to_sub_pos = None
        self.lines_or_to_sub_pos = None
        self.lines_ex_to_sub_pos = None

        # position in the topological vector
        # for internal use only, set it with "_compute_pos_big_topo"
        # after having loaded the _grid.
        self.load_pos_topo_vect = None
        self.gen_pos_topo_vect = None
        self.lines_or_pos_topo_vect = None
        self.lines_ex_pos_topo_vect = None
        # see definition of "_compute_pos_big_topo" for more information about these vector

        # the power _grid manipulated. One powergrid per backend.
        self._grid = None
        self.name_loads = None
        self.name_prods = None
        self.name_lines = None
        self.name_subs = None

        # thermal limit setting, in ampere, at the same "side" of the powerline than self.get_line_overflow
        self.thermal_limit_a = None

    def get_lines_id(self, _sentinel=None, from_=None, to_=None):
        """
        Returns the list of all the powerlines id in the backend going from "from_" to "to_"

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
            raise BackendError("Backend.get_lines_id: impossible to look for a powerline with no origin substation. Please modify \"from_\" parameter")
        if to_ is None:
            raise BackendError("Backend.get_lines_id: impossible to look for a powerline with no extremity substation. Please modify \"to_\" parameter")

        for i, (ori, ext) in enumerate(zip(self.lines_or_to_subid, self.lines_ex_to_subid)):
            if ori == from_ and ext == to_:
                res.append(i)

        if res is []:
            raise BackendError("Backend.get_line_id: impossible to find a powerline with connected at origin at {} and extremity at {}".format(from_, to_))

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
                "Backend.get_generators_id: impossible to look for a generator not connected to any substation. Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.gen_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "Backend.get_generators_id: impossible to find a generator connected at substation {}".format(sub_id))

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
                "Backend.get_loads_id: impossible to look for a load not connected to any substation. Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.load_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "Backend.get_loads_id: impossible to find a load connected at substation {}".format(sub_id))

        return res

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
            obj_before = np.sum(self.subs_elements[:sub_id])
            res[i] = obj_before + my_pos
        return res

    def _compute_pos_big_topo(self):
        """
        Compute the position of each element in the big topological vector.

        Topology action are represented by numpy vector of size np.sum(self.subs_elements).
        The vector self._load_pos_topo_vect will give the index of each load in this big topology vector.
        For examaple, for load i, self._load_pos_topo_vect[i] gives the position in such a topology vector that
        affect this load.

        This position can be automatically deduced from self.subs_elements, self._load_to_subid and self._load_to_sub_pos.

        This is the same for generators and both end of powerlines

        :return: ``None``
        """
        # self.assert_grid_correct()
        self.load_pos_topo_vect = self._aux_pos_big_topo(self.load_to_subid, self.load_to_sub_pos).astype(np.int)
        self.gen_pos_topo_vect = self._aux_pos_big_topo(self.gen_to_subid, self.gen_to_sub_pos).astype(np.int)
        self.lines_or_pos_topo_vect = self._aux_pos_big_topo(self.lines_or_to_subid, self.lines_or_to_sub_pos).astype(np.int)
        self.lines_ex_pos_topo_vect = self._aux_pos_big_topo(self.lines_ex_to_subid, self.lines_ex_to_sub_pos).astype(np.int)

    def assert_grid_correct(self):
        """
        Performs some checking on the loaded _grid to make sure it is consistent.
        It also makes sure that the vector such as *subs_elements*, *_load_to_subid* or *_gen_to_sub_pos* are of the
        right type eg. numpy.array with dtype: np.int

        It is called after the _grid has been loaded.

        These function is by default called by the :class:`grid2op.Environment` class after the initialization of the environment.
        If these tests are not successfull, no guarantee are given that the backend will return consistent computations.

        In order for the backend to fully understand the structure of actions, it is strongly advised NOT to override this method.

        :return: ``None``
        :raise: :class:`grid2op.EnvError` and possibly all of its derived class.
        """

        if self.name_lines is None:
            raise EnvError("name_lines is None. Backend is invalid. Line names are used to make the correspondance between the chronics and the backend")
        if self.name_loads is None:
            raise EnvError("name_loads is None. Backend is invalid. Line names are used to make the correspondance between the chronics and the backend")
        if self.name_prods is None:
            raise EnvError("name_prods is None. Backend is invalid. Line names are used to make the correspondance between the chronics and the backend")
        if self.name_subs is None:
            raise EnvError("name_subs is None. Backend is invalid. Line names are used to make the correspondance between the chronics and the backend")

        # test if vector can be properly converted
        if not isinstance(self.subs_elements, np.ndarray):
            try:
                self.subs_elements = np.array(self.subs_elements)
                self.subs_elements = self.subs_elements.astype(np.int)
            except Exception as e:
                raise EnvError("self.subs_elements should be convertible to a numpy array")

        if not isinstance(self.load_to_subid, np.ndarray):
            try:
                self.load_to_subid = np.array(self.load_to_subid)
                self.load_to_subid = self.load_to_subid.astype(np.int)
            except Exception as e:
                raise EnvError("self._load_to_subid should be convertible to a numpy array")
        if not isinstance(self.gen_to_subid, np.ndarray):
            try:
                self.gen_to_subid = np.array(self.gen_to_subid)
                self.gen_to_subid = self.gen_to_subid.astype(np.int)
            except Exception as e:
                raise EnvError("self._gen_to_subid should be convertible to a numpy array")
        if not isinstance(self.lines_or_to_subid, np.ndarray):
            try:
                self.lines_or_to_subid = np.array(self.lines_or_to_subid)
                self.lines_or_to_subid = self.lines_or_to_subid .astype(np.int)
            except Exception as e:
                raise EnvError("self._lines_or_to_subid should be convertible to a numpy array")
        if not isinstance(self.lines_ex_to_subid, np.ndarray):
            try:
                self.lines_ex_to_subid = np.array(self.lines_ex_to_subid)
                self.lines_ex_to_subid = self.lines_ex_to_subid.astype(np.int)
            except Exception as e:
                raise EnvError("self._lines_ex_to_subid should be convertible to a numpy array")

        if not isinstance(self.load_to_sub_pos, np.ndarray):
            try:
                self.load_to_sub_pos = np.array(self.load_to_sub_pos)
                self.load_to_sub_pos = self.load_to_sub_pos.astype(np.int)
            except Exception as e:
                raise EnvError("self._load_to_sub_pos should be convertible to a numpy array")
        if not isinstance(self.gen_to_sub_pos, np.ndarray):
            try:
                self.gen_to_sub_pos = np.array(self.gen_to_sub_pos)
                self.gen_to_sub_pos = self.gen_to_sub_pos.astype(np.int)
            except Exception as e:
                raise EnvError("self._gen_to_sub_pos should be convertible to a numpy array")
        if not isinstance(self.lines_or_to_sub_pos, np.ndarray):
            try:
                self.lines_or_to_sub_pos = np.array(self.lines_or_to_sub_pos)
                self.lines_or_to_sub_pos = self.lines_or_to_sub_pos.astype(np.int)
            except Exception as e:
                raise EnvError("self._lines_or_to_sub_pos should be convertible to a numpy array")
        if not isinstance(self.lines_ex_to_sub_pos, np.ndarray):
            try:
                self.lines_ex_to_sub_pos = np.array(self.lines_ex_to_sub_pos)
                self.lines_ex_to_sub_pos = self.lines_ex_to_sub_pos .astype(np.int)
            except Exception as e:
                raise EnvError("self._lines_ex_to_sub_pos should be convertible to a numpy array")

        if not isinstance(self.load_pos_topo_vect, np.ndarray):
            try:
                self.load_pos_topo_vect = np.array(self.load_pos_topo_vect)
                self.load_pos_topo_vect = self.load_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self._load_pos_topo_vect should be convertible to a numpy array")
        if not isinstance(self.gen_pos_topo_vect, np.ndarray):
            try:
                self.gen_pos_topo_vect = np.array(self.gen_pos_topo_vect)
                self.gen_pos_topo_vect = self.gen_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self._gen_pos_topo_vect should be convertible to a numpy array")
        if not isinstance(self.lines_or_pos_topo_vect, np.ndarray):
            try:
                self.lines_or_pos_topo_vect = np.array(self.lines_or_pos_topo_vect)
                self.lines_or_pos_topo_vect = self.lines_or_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self._lines_or_pos_topo_vect should be convertible to a numpy array")
        if not isinstance(self.lines_ex_pos_topo_vect, np.ndarray):
            try:
                self.lines_ex_pos_topo_vect = np.array(self.lines_ex_pos_topo_vect)
                self.lines_ex_pos_topo_vect = self.lines_ex_pos_topo_vect.astype(np.int)
            except Exception as e:
                raise EnvError("self._lines_ex_pos_topo_vect should be convertible to a numpy array")

        # test that all numbers are finite:
        tmp = np.concatenate((
            self.subs_elements.flatten(),
                             self.load_to_subid.flatten(),
                             self.gen_to_subid.flatten(),
                             self.lines_or_to_subid.flatten(),
                             self.lines_ex_to_subid.flatten(),
                             self.load_to_sub_pos.flatten(),
                             self.gen_to_sub_pos.flatten(),
                             self.lines_or_to_sub_pos.flatten(),
                             self.lines_ex_to_sub_pos.flatten(),
                             self.load_pos_topo_vect.flatten(),
                             self.gen_pos_topo_vect.flatten(),
                             self.lines_or_pos_topo_vect.flatten(),
                             self.lines_ex_pos_topo_vect.flatten()
                              ))
        try:
            if np.any(~np.isfinite(tmp)):
                raise EnvError("One of the vector is made of non finite elements")
        except Exception as e:
            raise EnvError("Impossible to check wheter or not vectors contains online finite elements (pobably one or more topology related vector is not valid (None)")

        # check sizes
        if len(self.subs_elements) != self.n_substations:
            raise IncorrectNumberOfSubstation("The number of substation is not consistent in self.subs_elements (size \"{}\") and  self.n_substations ({})".format(len(self.subs_elements), self.n_substations))
        if np.sum(self.subs_elements) != self.n_loads + self.n_generators + 2*self.n_lines:
            err_msg = "The number of elements of elements is not consistent between self.subs_elements where there are "
            err_msg +=  "{} elements connected to all substations and the number of load, generators and lines in the _grid."
            err_msg = err_msg.format(np.sum(self.subs_elements))
            raise IncorrectNumberOfElements(err_msg)

        if len(self.load_to_subid) != self.n_loads:
            raise IncorrectNumberOfLoads()
        if len(self.gen_to_subid) != self.n_generators:
            raise IncorrectNumberOfGenerators()
        if len(self.lines_or_to_subid) != self.n_lines:
            raise IncorrectNumberOfLines()
        if len(self.lines_ex_to_subid) != self.n_lines:
            raise IncorrectNumberOfLines()

        if len(self.load_to_sub_pos) != self.n_loads:
            raise IncorrectNumberOfLoads()
        if len(self.gen_to_sub_pos) != self.n_generators:
            raise IncorrectNumberOfGenerators()
        if len(self.lines_or_to_sub_pos) != self.n_lines:
            raise IncorrectNumberOfLines()
        if len(self.lines_ex_to_sub_pos) != self.n_lines:
            raise IncorrectNumberOfLines()

        if len(self.load_pos_topo_vect) != self.n_loads:
            raise IncorrectNumberOfLoads()
        if len(self.gen_pos_topo_vect) != self.n_generators:
            raise IncorrectNumberOfGenerators()
        if len(self.lines_or_pos_topo_vect) != self.n_lines:
            raise IncorrectNumberOfLines()
        if len(self.lines_ex_pos_topo_vect) != self.n_lines:
            raise IncorrectNumberOfLines()

        # test if object are connected to right substation
        obj_per_sub = np.zeros(shape=(self.n_substations,))
        for sub_id in self.load_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in self.gen_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in self.lines_or_to_subid:
            obj_per_sub[sub_id] += 1
        for sub_id in self.lines_ex_to_subid:
            obj_per_sub[sub_id] += 1

        if not np.all(obj_per_sub == self.subs_elements):
            raise IncorrectNumberOfElements()

        # test right number of element in substations
        # test that for each substation i don't have an id above the number of element of a substations
        for i, (sub_id, sub_pos) in enumerate(zip(self.load_to_subid, self.load_to_sub_pos)):
            if sub_pos >= self.subs_elements[sub_id]:
                raise IncorrectPositionOfLoads("for load {}".format(i))
        for i, (sub_id, sub_pos) in enumerate(zip(self.gen_to_subid, self.gen_to_sub_pos)):
            if sub_pos >= self.subs_elements[sub_id]:
                raise IncorrectPositionOfGenerators("for generator {}".format(i))
        for i, (sub_id, sub_pos) in enumerate(zip(self.lines_or_to_subid, self.lines_or_to_sub_pos)):
            if sub_pos >= self.subs_elements[sub_id]:
                raise IncorrectPositionOfLines("for line {} at origin end".format(i))
        for i, (sub_id, sub_pos) in enumerate(zip(self.lines_ex_to_subid, self.lines_ex_to_sub_pos)):
            if sub_pos >= self.subs_elements[sub_id]:
                # pdb.set_trace()
                raise IncorrectPositionOfLines("for line {} at extremity end".format(i))

        # check that i don't have 2 objects with the same id in the "big topo" vector
        if len(np.unique(np.concatenate((self.load_pos_topo_vect.flatten(),
                                        self.gen_pos_topo_vect.flatten(),
                                        self.lines_or_pos_topo_vect.flatten(),
                                        self.lines_ex_pos_topo_vect.flatten())))) != np.sum(self.subs_elements):
                raise EnvError("2 different objects would have the same id in the topology vector.")

        # check that self._load_pos_topo_vect and co are consistent
        load_pos_big_topo = self._aux_pos_big_topo(self.load_to_subid, self.load_to_sub_pos)
        if not np.all(load_pos_big_topo == self.load_pos_topo_vect):
            raise IncorrectPositionOfLoads()
        gen_pos_big_topo = self._aux_pos_big_topo(self.gen_to_subid, self.gen_to_sub_pos)
        if not np.all(gen_pos_big_topo == self.gen_pos_topo_vect):
            raise IncorrectNumberOfGenerators()
        lines_or_pos_big_topo = self._aux_pos_big_topo(self.lines_or_to_subid, self.lines_or_to_sub_pos)
        if not np.all(lines_or_pos_big_topo == self.lines_or_pos_topo_vect):
            raise IncorrectPositionOfLines()
        lines_ex_pos_big_topo = self._aux_pos_big_topo(self.lines_ex_to_subid, self.lines_ex_to_sub_pos)
        if not np.all(lines_ex_pos_big_topo == self.lines_ex_pos_topo_vect):
            raise IncorrectPositionOfLines()

        # no empty bus: at least one element should be present on each bus
        if np.any(self.subs_elements < 1):
            raise BackendError("There are {} bus with 0 element connected to it.".format(np.sum(self.subs_elements < 1)))

    def assert_grid_correct_after_powerflow(self):
        """
        This method is called by the environment. It ensure that the backend remains consistent even after a powerflow has be run with :func:`Backend.runpf` method.

        :return: ``None``
        :raise: :class:`grid2op.Exceptions.EnvError` and possibly all of its derived class.
        """
        # test the results gives the proper size
        tmp = self.get_line_status()
        if tmp.shape[0] != self.n_lines:
            raise IncorrectNumberOfLines("returned by \"backend.get_line_status()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please your data.")
        tmp = self.get_line_flow()
        if tmp.shape[0] != self.n_lines:
            raise IncorrectNumberOfLines("returned by \"backend.get_line_flow()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please your data.")
        tmp = self.get_thermal_limit()
        if tmp.shape[0] != self.n_lines:
            raise IncorrectNumberOfLines("returned by \"backend.get_thermal_limit()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please your data.")
        tmp = self.get_line_overflow()
        if tmp.shape[0] != self.n_lines:
            raise IncorrectNumberOfLines("returned by \"backend.get_line_overflow()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please your data.")

        tmp = self.generators_info()
        if len(tmp) != 3:
            raise EnvError("\"generators_info()\" should return a tuple with 3 elements: p, q and v")
        for el in tmp:
            if el.shape[0] != self.n_generators:
                raise IncorrectNumberOfGenerators("returned by \"backend.generators_info()\"")
        tmp = self.loads_info()
        if len(tmp) != 3:
            raise EnvError("\"loads_info()\" should return a tuple with 3 elements: p, q and v")
        for el in tmp:
            if el.shape[0] != self.n_loads:
                raise IncorrectNumberOfLoads("returned by \"backend.loads_info()\"")
        tmp = self.lines_or_info()
        if len(tmp) != 4:
            raise EnvError("\"lines_or_info()\" should return a tuple with 4 elements: p, q, v and a")
        for el in tmp:
            if el.shape[0] != self.n_lines:
                raise IncorrectNumberOfLines("returned by \"backend.lines_or_info()\"")
        tmp = self.lines_ex_info()
        if len(tmp) != 4:
            raise EnvError("\"lines_ex_info()\" should return a tuple with 4 elements: p, q, v and a")
        for el in tmp:
            if el.shape[0] != self.n_lines:
                raise IncorrectNumberOfLines("returned by \"backend.lines_ex_info()\"")

        tmp = self.get_topo_vect()
        if tmp.shape[0] != np.sum(self.subs_elements):
            raise IncorrectNumberOfElements("returned by \"backend.get_topo_vect()\"")

        if np.any(~np.isfinite(tmp)):
            raise EnvError("Some components of \"backend.get_topo_vect()\" are not finite. This should be integer.")

    @abstractmethod
    def load_grid(self, path, filename=None):
        """
        Load the powergrid.
        It should first define self._grid.
        And then fill all the helpers used by the backend eg. :attr:`Backend._n_lines`, :attr:`Backend.subs_elements`, :attr:`Backend._load_to_subid`,
        :attr:`Backend._gen_to_sub_pos` or :attr:`Backend._lines_ex_pos_topo_vect`.

        After a the call to :func:`Backend.load_grid` has been performed, the backend should be in such a state that

        :param path: the path to find the powergrid
        :type path: :class:`string`

        :param filename: the filename of the powergrid
        :type filename: :class:`string`, optional

        :return: ``None``
        """
        pass

    @abstractmethod
    def close(self):
        """
        This function is called when the environment is over.
        After calling this function, the backend might not behave properly, and in any case should not be used before
        another call to :func:`Backend.load_grid` is performed
        Returns
        -------
        ``None``
        """

    @abstractmethod
    def apply_action(self, action):
        """
        Modify the powergrid with the action given by an agent or by the envir.
        For the L2RPN project, this action is mainly for topology if it has been sent by the agent.
        Or it can also affect production and loads, if the action is made by the environment.

        The help of :class:`grid2op.Action` or the code in Action.py file give more information about the implementation of this method.

        :param action: the action to be implemented on the powergrid.
        :type action: :class:`grid2op.Action.Action`

        :return: ``None``
        """
        pass

    @abstractmethod
    def runpf(self, is_dc=False):
        """
        Run a power flow on the underlying _grid.
        Powerflow can be AC (is_dc = False) or DC (is_dc = True)

        :param is_dc: is the powerflow run in DC or in AC
        :type is_dc: :class:`bool`

        :return: True if it has converged, or false otherwise. In case of non convergence, no flows can be inspected on the _grid.
        :rtype: :class:`bool`
        """
        pass

    @abstractmethod
    def copy(self):
        """
        Performs a deep copy of the backend.

        :return: An instance of Backend equal to :attr:`.self`, but deep copied.
        :rtype: :class:`Backend`
        """
        pass

    def save_file(self, full_path):
        """
        Save the current power _grid in a human readable format supported by the backend.
        The format is not modified by this wrapper.

        This function is not mandatory, and if implemented, it is used only as a debugging purpose.

        :param full_path: the full path (path + file name + extension) where *self._grid* is stored.
        :type full_path: :class:`string`

        :return: ``None``
        """
        raise RuntimeError("Class {} does not allow for saving file.".format(self))

    @abstractmethod
    def get_line_status(self):
        """
        Return the status of each lines (connected : True / disconnected: False )

        It is assume that the order of the powerline is fixed: if the status of powerline "l1" is put at the 42nd element
        of the return vector, then it should always be set at the 42nd element.

        It is also assumed that all the other methods of the backend that allows to retrieve informations on the powerlines
        also respect the same convention, and consistent with one another.
        For example, if powerline "l1" is the 42nd second of the vector returned by :func:`Backend.get_line_status` then information
        about it's flow will be at position *42* of the vector returned by :func:`Backend.get_line_flow` for example.

        :return: an array with the line status of each powerline
        :rtype: np.array, dtype:bool
        """
        pass

    @abstractmethod
    def get_line_flow(self):
        """
        Return the current flow in each lines of the powergrid. Only one value per powerline is returned.

        If the AC mod is used, this shall return the current flow on the end of the powerline where there is a protection.
        For example, if there is a protection on "origin end" of powerline "l2" then this method shall return the current
        flow of at the "origin end" of powerline l2.

        Note that in general, there is no loss of generality in supposing all protections are set on the "origin end" of
        the powerline. So this method will return all origin line flows.
        It is also possible, for a specific application, to return the maximum current flow between both ends of a power
        _grid for more complex scenario.

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        :return: an array with the line flows of each powerline
        :rtype: np.array, dtype:float
        """
        pass

    def set_thermal_limit(self, limits):
        """
        This function is used as a convenience function to set the thermal limits :attr:`Backend.thermal_limit_a`
        in amperes.

        It can be used at the beginning of an episode if the thermal limit are not present in the original data files
        or alternatively if the thermal limits depends on the period of the year (one in winter and one in summer
        for example).

        Parameters
        ----------
        limits: ``object``
            It can be understood differently according to its type:

            - If it's a ``numpy.ndarray``, then it is assumed the thermal limits are given in amperes in the same order
              as the powerlines computed in the backend. In that case it modifies all the thermal limits of all
              the powerlines at once.
            - If it's a ``dict`` it must have:

              - as key the powerline names (not all names are mandatory, in that case only the powerlines with the name
                in this dictionnary will be modified)
              - as value the new thermal limit (should be a strictly positive float).


        Returns
        -------
        ``None``

        """
        if isinstance(limits, np.ndarray):
            if limits.shape[0] == self.n_lines:
                self.thermal_limit_a = 1. * limits
        elif isinstance(limits, dict):
            for el in limits.keys():
                if not el in self.name_lines:
                    raise BackendError("You asked to modify the thermal limit of powerline named \"{}\" that is not on the grid. Names of powerlines are {}".format(el, self.name_lines))
            for i, el in self.name_lines:
                if el in limits:
                    try:
                        tmp = float(limits[el])
                    except:
                        raise BackendError("Impossible to convert data ({}) for powerline named \"{}\" into float values".format(limits[el], el))
                    if tmp <= 0:
                        raise BackendError("New thermal limit for powerlines \"{}\" is not positive ({})".format(el, tmp))
                    self.thermal_limit_a[i] = tmp

    def update_thermal_limit(self, env):
        """
        Upade the new thermal limit in case of DLR for example.

        By default it does nothing.

        Depending on the operational strategy, it is also possible to implement some
        `Dynamic Line Rating <https://en.wikipedia.org/wiki/Dynamic_line_rating_for_electric_utilities>`_ (DLR)
        strategies.
        In this case, this function will give the thermal limit for a given time step provided the flows and the
        weather condition are accessible by the backend. Our methodology doesn't make any assumption on the method
        used to get these thermal limits.


        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment used to compute the thermal limit

        Returns
        -------
        ``None``
        """

        pass

    def get_thermal_limit(self):
        """
        Gives the thermal limit (in amps) for each powerline of the _grid. Only one value per powerline is returned.

        It is assumed that both :func:`Backend.get_line_flow` and *_get_thermal_limit* gives the value of the same end of the powerline.
        See the help of *_get_line_flow* for a more detailed description of this problem.

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        :return: An array giving the thermal limit of the powerlines.
        :rtype: np.array, dtype:float
        """
        return self.thermal_limit_a

    def get_relative_flow(self):
        """
        This method return the relative flows, *eg.* the current flow divided by the thermal limits. It has a pretty
        straightforward default implementation, but it can be overriden for example for transformer if the limits are
        on the lower voltage side or on the upper voltage level.

        Returns
        -------
        res: ``numpy.ndarray``, dtype: float
            The relative flow in each powerlines of the grid.
        """
        num_ = self.get_line_flow()
        denom_ = self.get_thermal_limit()
        return num_ / denom_

    def get_line_overflow(self):
        """
        faster accessor to the line that are on overflow.

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        :param obs: the current observation in which the powerflow is done. This can be use for example in case of DLR implementation.
        :type obs: Observation

        :return: An array saying if a powerline is overflow or not
        :rtype: np.array, dtype:bool
        """
        th_lim = self.get_thermal_limit()
        flow = self.get_line_flow()
        return flow > th_lim

    @abstractmethod
    def get_topo_vect(self):
        """
        Get the topology vector from the _grid.
        The topology vector defines, for each object, on which bus it is connected.
        It returns -1 if the object is not connected.

        :return: An array saying to which bus the object is connected.
        :rtype: np.array, dtype:bool
        """
        pass

    @abstractmethod
    def generators_info(self):
        """
        This method is used to retrieve informations about the generators.

        Returns
        -------
        prod_p ``numpy.array``
            The active power production for each generator
        prod_q ``numpy.array``
            The reactive power production for each generator
        prod_v ``numpy.array``
            The voltage magnitude of the bus to which each generators is connected
        """
        pass

    @abstractmethod
    def loads_info(self):
        """
        This method is used to retrieve informations about the loads.

        Returns
        -------
        load_p ``numpy.array``
            The active power consumption for each load
        load_q ``numpy.array``
            The reactive power consumption for each load
        load_v ``numpy.array``
            The voltage magnitude of the bus to which each load is connected
        """
        pass

    @abstractmethod
    def lines_or_info(self):
        """
        It returns the information extracted from the _grid at the origin end of each powerline.

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        Returns
        -------
        p_or ``numpy.array``
            the origin active power flowing on the lines
        q_or ``numpy.array``
            the origin reactive power flowing on the lines
        v_or ``numpy.array``
            the voltage magnitude at the origin of each powerlines
        a_or ``numpy.array``
            the current flow at the origin of each powerlines
        """
        pass

    @abstractmethod
    def lines_ex_info(self):
        """
        It returns the information extracted from the _grid at the extremity end of each powerline.

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        Returns
        -------
        p_ex ``numpy.array``
            the extremity active power flowing on the lines
        q_ex ``numpy.array``
            the extremity reactive power flowing on the lines
        v_ex ``numpy.array``
            the voltage magnitude at the extremity of each powerlines
        a_ex ``numpy.array``
            the current flow at the extremity of each powerlines
        """
        pass

    def shunt_info(self):
        """
        This method is optional. If implemented, it should return the proper information about the shunt in the powergrid.

        If not implemented it returns empty list.

        Note that if there are shunt on the powergrid, it is recommended that this method should be implemented before
        calling :func:`Backend.check_kirchoff`.

        If this method is implemented AND :func:`Backend.check_kirchoff` is called, the method
        :func:`Backend.sub_from_bus_id` should also be implemented preferably.

        Returns
        -------
        shunt_p ``numpy.array``
            For each shunt, the active power it withdraw at the bus to which it is connected.
        shunt_q ``numpy.array``
            For each shunt, the reactive power it withdraw at the bus to which it is connected.
        shunt_v ``numpy.array``
            For each shunt, the voltage magnitude of the bus to which it is connected.
        shunt_bus ``numpy.array``
            For each shunt, the bus id to which it is connected.
        """
        return [], [], [], []

    def sub_from_bus_id(self, bus_id):
        """
        Optionnal method that allows to get the substation if the bus id is provided.

        :param bus_id:
        :return: the substation to which an object connected to bus with id `bus_id` is connected to.
        """
        raise Grid2OpException("This backend doesn't allow to get the substation from the bus id.")

    @abstractmethod
    def _disconnect_line(self, id):
        """
        Disconnect the line of id "id" in the backend.
        In this scenario, the *id* of a powerline is its position (counted starting from O) in the vector returned by
        :func:`Backend.get_line_status` or :func:`Backend.get_line_flow` for example.
        For example, if the current flow on powerline "l1" is the 42nd element of the vector returned by :func:`Backend.get_line_flow`
        then :func:`Backend._disconnect_line(42)` will disconnect this same powerline "l1".

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        :param id: id of the powerline to be disconnected
        :type id: int

        :return: ``None``
        """
        pass

    def _runpf_with_diverging_exception(self, is_dc):
        """
        Computes a power flow on the _grid and raises an exception in case of diverging power flow, or any other
        exception that can be thrown by the backend.

        :param is_dc: mode of the power flow. If *is_dc* is True, then the powerlow is run using the DC approximation otherwise it uses the AC powerflow.
        :type is_dc: bool

        :return: ``None``
        """
        conv = False
        try:
            conv = self.runpf(is_dc=is_dc)  # run powerflow
        except:
            pass

        if not conv:
            raise DivergingPowerFlow("Powerflow has diverged during computation.")

    def next_grid_state(self, env, is_dc=False):
        """
        This method is called by the environment to compute the next _grid states.
        It allows to compute the powerline and approximate the "cascading failures" if there are some overflows.

        Note that it **DOESNT** update the environment with the disconnected lines.

        :param env: the environment in which the powerflow is ran.
        :type env: :class:`grid2op.Environment.Environment`

        :param is_dc: mode of power flow (AC : False, DC: is_dc is True)
        :type is_dc: bool

        :return: disconnected lines and list of Backend instances that allows to reconstruct the cascading failures (in which order the powerlines have been disconnected). Note that if :attr:`Backend.detailed_infos_for_cascading_failures` is set to False, the empty list will always be returned.
        :rtype: tuple: np.array, dtype:bool, list
        """

        lines_status_orig = self.get_line_status()  # original line status
        infos = []
        self._runpf_with_diverging_exception(is_dc)

        disconnected_during_cf = np.full(self.n_lines, fill_value=False, dtype=np.bool)
        if env.no_overflow_disconnection:
            return disconnected_during_cf, infos

        # the environment disconnect some
        init_time_step_overflow = copy.deepcopy(env.timestep_overflow)
        while True:
            # simulate the cascading failure
            lines_flows = self.get_line_flow()
            thermal_limits = self.get_thermal_limit()
            lines_status = self.get_line_status()

            # a) disconnect lines on hard overflow
            to_disc = lines_flows > env.hard_overflow_threshold * thermal_limits

            # b) deals with soft overflow
            init_time_step_overflow[ (lines_flows >= thermal_limits) & (lines_status)] += 1
            to_disc[init_time_step_overflow > env.nb_timestep_overflow_allowed] = True

            # disconnect the current power lines
            if np.sum(to_disc[lines_status]) == 0:
                # no powerlines have been disconnected at this time step, i stop the computation there
                break
            disconnected_during_cf[to_disc] = True
            # perform the disconnection action
            [self._disconnect_line(i) for i, el in enumerate(to_disc) if el]

            # start a powerflow on this new state
            self._runpf_with_diverging_exception(self._grid)
            if self.detailed_infos_for_cascading_failures:
                infos.append(self.copy())
        return disconnected_during_cf, infos

    def check_kirchoff(self):
        """
        Check that the powergrid respects kirchhoff's law.
        This function can be called at any moment to make sure a powergrid is in a consistent state, or to perform
        some tests for example.

        In order to function properly, this method requires that :func:`Backend.shunt_info` and
        :func:`Backend.sub_from_bus_id` are properly defined. Otherwise the results might be wrong, especially
        for reactive values (q_subs and q_bus bellow)

        Returns
        -------
        p_subs ``numpy.array``
            sum of injected active power at each substations
        q_subs ``numpy.array``
            sum of injected reactive power at each substations
        p_bus ``numpy.array``
            sum of injected active power at each buses. It is given in form of a matrix, with number of substations as
            row, and number of columns equal to the maximum number of buses for a substation
        q_bus ``numpy.array``
            sum of injected reactive power at each buses. It is given in form of a matrix, with number of substations as
            row, and number of columns equal to the maximum number of buses for a substation
        """

        p_or, q_or, v_or, *_ = self.lines_or_info()
        p_ex, q_ex, v_ex, *_ = self.lines_ex_info()
        p_gen, q_gen, v_gen = self.generators_info()
        p_load, q_load, v_load = self.loads_info()
        p_s, q_s, v_s, bus_s = self.shunt_info()

        try:
            self.sub_from_bus_id(0)
            can_extract_shunt = True
        except:
            can_extract_shunt = False

        # fist check the "substation law" : nothing is created at any substation
        p_subs = np.zeros(self.n_substations)
        q_subs = np.zeros(self.n_substations)

        # check for each bus
        p_bus = np.zeros((self.n_substations, 2))
        q_bus = np.zeros((self.n_substations, 2))
        topo_vect = self.get_topo_vect()

        for i in range(self.n_lines):
            # for substations
            p_subs[self.lines_or_to_subid[i]] += p_or[i]
            p_subs[self.lines_ex_to_subid[i]] += p_ex[i]

            q_subs[self.lines_or_to_subid[i]] += q_or[i]
            q_subs[self.lines_ex_to_subid[i]] += q_ex[i]

            # for bus
            p_bus[self.lines_or_to_subid[i], topo_vect[self.lines_or_pos_topo_vect[i]]-1] += p_or[i]
            q_bus[self.lines_or_to_subid[i], topo_vect[self.lines_or_pos_topo_vect[i]]-1] += q_or[i]

            p_bus[self.lines_ex_to_subid[i], topo_vect[self.lines_ex_pos_topo_vect[i]]-1] += p_ex[i]
            q_bus[self.lines_ex_to_subid[i], topo_vect[self.lines_ex_pos_topo_vect[i]]-1] += q_ex[i]

        for i in range(self.n_generators):
            # for substations
            p_subs[self.gen_to_subid[i]] -= p_gen[i]
            q_subs[self.gen_to_subid[i]] -= q_gen[i]

            # for bus
            p_bus[self.gen_to_subid[i],  topo_vect[self.gen_pos_topo_vect[i]]-1] -= p_gen[i]
            q_bus[self.gen_to_subid[i],  topo_vect[self.gen_pos_topo_vect[i]]-1] -= q_gen[i]

        for i in range(self.n_loads):
            # for substations
            p_subs[self.load_to_subid[i]] += p_load[i]
            q_subs[self.load_to_subid[i]] += q_load[i]

            # for buses
            p_bus[self.load_to_subid[i],  topo_vect[self.load_pos_topo_vect[i]]-1] += p_load[i]
            q_bus[self.load_to_subid[i],  topo_vect[self.load_pos_topo_vect[i]]-1] += q_load[i]

        if can_extract_shunt:
            for i in range(len(p_s)):
                tmp_bus = bus_s[i]
                sub_id = self.sub_from_bus_id(tmp_bus)
                p_subs[sub_id] += p_s[i]
                q_subs[sub_id] += q_s[i]

                p_bus[sub_id, 1*(tmp_bus!=sub_id)] += p_s[i]
                q_bus[sub_id, 1*(tmp_bus!=sub_id)] += q_s[i]
        else:
            warnings.warn("Backend.check_kirchoff Impossible to get shunt information. Reactive information might be incorrect.")

        return p_subs, q_subs, p_bus, q_bus





