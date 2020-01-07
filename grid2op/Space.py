"""
This class abstract the main components of Action, Observation, ActionSpace and ObservationSpace.

It basically represents a powergrid (the object in it) in a format completely agnostic to the solver use to compute
the powerflows (:class:grid2op.Backend.Backend).

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
# TODO gym integration (inheritance, shape and dtype)
# TODO tests of these methods
class GridObjects:
    """
    This class stores in a Backend agnostic way some information about the powergrid.

    It stores information about number of objects, and which objects are where, their names etc.

    The classes :class:`grid2op.Action.Action`, :class:`grid2op.Action.HelperAction`,
    :class:`grid2op.Observation.Observation`, :class:`grid2op.Observation.ObservationHelper` and
    :class:`grid2op.Backend.Backend` all inherit from this class. This means that each of the above has its own
    representation of the powergrid.

    **NB** it does not store any information about the current state of the powergrid. It stores information that
    cannot be modified by the Agent, the Environment or any other entity.

    Attributes
    ----------

    n_line: :class:`int`
        number of powerline in the _grid

    n_gen: :class:`int`
        number of generators in the _grid

    n_load: :class:`int`
        number of loads in the powergrid

    sub_info: :class:`numpy.array`, dtype:int
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
        :attr:`GridObjects.sub_info`\[i\]. For a given load of id *l*,
        :attr:`Action.GridObjects.load_to_sub_pos`\[l\] is the index
        of the load *l* in the vector *sub_topo_vect*. This means that, if
        *sub_topo_vect\[ action.load_to_sub_pos\[l\] \]=2*
        then load of id *l* is connected to the second bus of the substation.

    gen_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for generators.

    lines_or_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos`  but for "origin" end of powerlines.

    lines_ex_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`GridObjects.load_to_sub_pos` but for "extremity" end of powerlines.

    load_pos_topo_vect: :class:`numpy.array`, dtype:int
        It has a similar role as :attr:`GridObjects.load_to_sub_pos` but it gives the position in the vector representing
        the whole topology. More concretely, if the complete topology of the powergrid is represented here by a vector
        *full_topo_vect* resulting of the concatenation of the topology vector for each substation
        (see :attr:`GridObjects.load_to_sub_pos`for more information). For a load of id *l* in the powergrid,
        :attr:`GridObjects.load_pos_topo_vect`\[l\] gives the index, in this *full_topo_vect* that concerns load *l*.
        More formally, if *_topo_vect\[ backend.load_pos_topo_vect\[l\] \]=2* then load of id l is connected to the
        second bus of the substation.

    gen_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for generators.

    lines_or_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for "origin" end of powerlines.

    lines_ex_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`GridObjects.load_pos_topo_vect` but for "extremity" end of powerlines.

    name_load: :class:`numpy.array`, dtype:str
        ordered name of the loads in the grid.

    name_prod: :class:`numpy.array`, dtype:str
        ordered name of the productions in the grid.

    name_line: :class:`numpy.array`, dtype:str
        ordered names of the powerline in the grid.

    name_sub: :class:`numpy.array`, dtype:str
        ordered names of the substation in the grid
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

    def init_grid_vect(self, name_prod, name_load, name_line, name_sub, sub_info,
                       load_to_subid, gen_to_subid, line_or_to_subid, line_ex_to_subid,
                       load_to_sub_pos, gen_to_sub_pos, line_or_to_sub_pos, line_ex_to_sub_pos,
                       load_pos_topo_vect, gen_pos_topo_vect, line_or_pos_topo_vect, line_ex_pos_topo_vect):
        """

        Parameters
        ----------
        name_prod: :class:`numpy.array`, dtype:str
            Used to initialized :attr:`GridObjects.name_gen`

        name_load: :class:`numpy.array`, dtype:str
            Used to initialized :attr:`GridObjects.name_load`

        name_line: :class:`numpy.array`, dtype:str
            Used to initialized :attr:`GridObjects.name_line`

        name_sub: :class:`numpy.array`, dtype:str
            Used to initialized :attr:`GridObjects.name_sub`

        sub_info: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.sub_info`

        load_to_subid: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.load_to_subid`

        gen_to_subid: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.gen_to_subid`

        lines_or_to_subid: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.line_or_to_subid`

        lines_ex_to_subid: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.line_ex_to_subid`

        load_to_sub_pos: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.load_to_sub_pos`

        gen_to_sub_pos: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.gen_to_sub_pos`

        lines_or_to_sub_pos: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.line_or_to_sub_pos`

        lines_ex_to_sub_pos: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.line_ex_to_sub_pos`

        load_pos_topo_vect: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.load_pos_topo_vect`

        gen_pos_topo_vect: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.gen_pos_topo_vect`

        line_or_pos_topo_vect: :class:`numpy.array`, dtype:int
            Used to initialized :attr:`GridObjects.line_or_pos_topo_vect`

        line_ex_pos_topo_vect: :class:`numpy.array`, dtype:int
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
        right type eg. numpy.array with dtype: np.int

        It is called after the _grid has been loaded.

        These function is by default called by the :class:`grid2op.Environment` class after the initialization of the environment.
        If these tests are not successfull, no guarantee are given that the backend will return consistent computations.

        In order for the backend to fully understand the structure of actions, it is strongly advised NOT to override this method.

        :return: ``None``
        :raise: :class:`grid2op.EnvError` and possibly all of its derived class.
        """

        if self.name_line is None:
            raise EnvError("name_line is None. Powergrid is invalid. Line names are used to make the correspondance between the chronics and the backend")
        if self.name_load is None:
            raise EnvError("name_load is None. Powergrid is invalid. Line names are used to make the correspondance between the chronics and the backend")
        if self.name_gen is None:
            raise EnvError("name_gen is None. Powergrid is invalid. Line names are used to make the correspondance between the chronics and the backend")
        if self.name_sub is None:
            raise EnvError("name_sub is None. Powergrid is invalid. Substation names are used to make the correspondance between the chronics and the backend")

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
            raise EnvError("Impossible to check wheter or not vectors contains online finite elements (pobably one or more topology related vector is not valid (None)")

        # check sizes
        if len(self.sub_info) != self.n_sub:
            raise IncorrectNumberOfSubstation("The number of substation is not consistent in self.sub_info (size \"{}\") and  self.n_sub ({})".format(len(self.sub_info), self.n_sub))
        if np.sum(self.sub_info) != self.n_load + self.n_gen + 2*self.n_line:
            err_msg = "The number of elements of elements is not consistent between self.sub_info where there are "
            err_msg +=  "{} elements connected to all substations and the number of load, generators and lines in the _grid."
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
            raise BackendError("ObservationSpace.get_lines_id: impossible to look for a powerline with no origin substation. Please modify \"from_\" parameter")
        if to_ is None:
            raise BackendError("ObservationSpace.get_lines_id: impossible to look for a powerline with no extremity substation. Please modify \"to_\" parameter")

        for i, (ori, ext) in enumerate(zip(self.line_or_to_subid, self.line_ex_to_subid)):
            if ori == from_ and ext == to_:
                res.append(i)

        if res is []:
            raise BackendError("ObservationSpace.get_line_id: impossible to find a powerline with connected at origin at {} and extremity at {}".format(from_, to_))

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
                "GridObjects.get_generators_id: impossible to look for a generator not connected to any substation. Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.gen_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "GridObjects.get_generators_id: impossible to find a generator connected at substation {}".format(sub_id))

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
                "GridObjects.get_loads_id: impossible to look for a load not connected to any substation. Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.load_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "GridObjects.get_loads_id: impossible to find a load connected at substation {}".format(sub_id))

        return res


class SerializableSpace(GridObjects):
    """
    This class allows to serialize / de serialize the action space or observation space.

    It should not be used inside an Environment, as some functions of the action might not be compatible with
    the serialization, especially the checking of whether or not an Action is legal or not.

    Attributes
    ----------

    subtype: ``type``
        Type use to build the template object :attr:`SerializableSpace.template_obj`. This type should derive
        from :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`.

    template_obj: [:class:`grid2op.Action.Action`, :class:`grid2op.Observation.Observation`]
        An instance of the "*actionClass*" provided used to provide higher level utilities, such as the size of the
        action (see :func:`Action.size`) or to sample a new Action (see :func:`Action.sample`)

    n: ``int``
        Size of the space

    space_prng: ``np.random.RandomState``
        The random state of the observation (in case of non deterministic observations. This should not be used at the
        moment)

    seed: ``int``
        The seed used throughout the episode in case of non deterministic observations.

    shape: ``None``
        For gym compatibility, do not use yet

    dtype: ``None``
        For gym compatibility, do not use yet

    """
    def __init__(self, gridobj,
                 subtype=object):
        """

        subtype: ``type``
            Type of action used to build :attr:`SerializableActionSpace.template_act`. This type should derive
            from :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`.

        """

        if not isinstance(subtype, type):
            raise Grid2OpException(
                "Parameter \"subtype\" used to build the Space should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(subtype)))

        GridObjects.__init__(self)
        self.init_grid(gridobj)

        self.subtype = subtype
        self.template_obj = self.subtype(gridobj=self)
        self.n = self.template_obj.size()

        self.space_prng = np.random.RandomState()
        self.seed = None

        self.global_vars = None

        # TODO
        self.shape = self.template_obj.shape()
        self.dtype = self.template_obj.dtype()

    def seed(self, seed):
        """
        Use to set the seed in case of non deterministic observations.
        :param seed:
        :return:
        """
        self.seed = seed
        if self.seed is not None:
            # in this case i have specific seed set. So i force the seed to be deterministic.
            self.space_prng.seed(seed=self.seed)

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

        name_prod = extract_from_dict(dict_, "name_gen", lambda x: np.array(x).astype(str))
        name_load = extract_from_dict(dict_, "name_load", lambda x: np.array(x).astype(str))
        name_line = extract_from_dict(dict_, "name_line", lambda x: np.array(x).astype(str))
        name_sub = extract_from_dict(dict_, "name_sub", lambda x: np.array(x).astype(str))

        sub_info = extract_from_dict(dict_, "sub_info", lambda x: np.array(x).astype(np.int))
        load_to_subid = extract_from_dict(dict_, "load_to_subid", lambda x: np.array(x).astype(np.int))
        gen_to_subid = extract_from_dict(dict_, "gen_to_subid", lambda x: np.array(x).astype(np.int))
        line_or_to_subid = extract_from_dict(dict_, "line_or_to_subid", lambda x: np.array(x).astype(np.int))
        line_ex_to_subid = extract_from_dict(dict_, "line_ex_to_subid", lambda x: np.array(x).astype(np.int))

        load_to_sub_pos = extract_from_dict(dict_, "load_to_sub_pos", lambda x: np.array(x).astype(np.int))
        gen_to_sub_pos = extract_from_dict(dict_, "gen_to_sub_pos", lambda x: np.array(x).astype(np.int))
        line_or_to_sub_pos = extract_from_dict(dict_, "line_or_to_sub_pos", lambda x: np.array(x).astype(np.int))
        line_ex_to_sub_pos = extract_from_dict(dict_, "line_ex_to_sub_pos", lambda x: np.array(x).astype(np.int))

        load_pos_topo_vect = extract_from_dict(dict_, "load_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        gen_pos_topo_vect = extract_from_dict(dict_, "gen_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        line_or_pos_topo_vect = extract_from_dict(dict_, "line_or_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        line_ex_pos_topo_vect = extract_from_dict(dict_, "line_ex_pos_topo_vect", lambda x: np.array(x).astype(np.int))

        actionClass_str = extract_from_dict(dict_, "subtype", str)
        actionClass_li = actionClass_str.split('.')

        # pdb.set_trace()
        if actionClass_li[-1] in globals():
            subtype = globals()[actionClass_li[-1]]
        else:
            # TODO make something better and recursive here
            exec("from {} import {}".format(".".join(actionClass_li[:-1]), actionClass_li[-1]))
            try:
                subtype = eval(actionClass_str)
            except NameError:
                if len(actionClass_li) > 1:
                    try:
                        subtype = eval(".".join(actionClass_li[1:]))
                    except:
                        msg_err_ = "Impossible to find the module \"{}\" to load back the space (ERROR 1). Try \"from {} import {}\""
                        raise Grid2OpException(msg_err_.format(actionClass_str, ".".join(actionClass_li[:-1]), actionClass_li[-1]))
                else:
                    msg_err_ = "Impossible to find the module \"{}\" to load back the space (ERROR 2). Try \"from {} import {}\""
                    raise Grid2OpException(msg_err_.format(actionClass_str, ".".join(actionClass_li[:-1]), actionClass_li[-1]))
            except AttributeError:
                try:
                    subtype = eval(actionClass_li[-1])
                except:
                    if len(actionClass_li) > 1:
                        msg_err_ = "Impossible to find the class named \"{}\" to load back the space (ERROR 3)" \
                                   "(module is found but not the class in it) Please import it via \"from {} import {}\"."
                        msg_err_ = msg_err_.format(actionClass_str,
                                                   ".".join(actionClass_li[:-1]),
                                                   actionClass_li[-1])
                    else:
                        msg_err_ = "Impossible to import the class named \"{}\" to load back the space (ERROR 4) (the " \
                                   "module is found but not the class in it)"
                        msg_err_ = msg_err_.format(actionClass_str)
                    raise Grid2OpException(msg_err_)

        gridobj = GridObjects()
        gridobj.init_grid_vect(name_prod, name_load, name_line, name_sub, sub_info,
                                load_to_subid, gen_to_subid, line_or_to_subid, line_ex_to_subid,
                                load_to_sub_pos, gen_to_sub_pos, line_or_to_sub_pos, line_ex_to_sub_pos,
                                load_pos_topo_vect, gen_pos_topo_vect, line_or_pos_topo_vect,
                                line_ex_pos_topo_vect)
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

    def from_vect(self, act):
        """
        Convert an action, represented as a vector to a valid :class:`Action` instance

        Parameters
        ----------
        act: ``numpy.ndarray``
            A object living in a space represented as a vector (typically an :class:`grid2op.Action.Action` or an
            :class:`grid2op.Observation.Observation` represented as a numpy vector)

        Returns
        -------
        res: :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`
            The corresponding action (or observation) as an object (and not as a vector). The return type is given
            by the type of :attr:`SerializableSpace.template_obj`

        """
        res = copy.deepcopy(self.template_obj)
        res.from_vect(act)
        return res
