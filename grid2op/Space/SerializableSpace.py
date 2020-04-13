# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import re
import json
import copy
import pdb

from grid2op.Exceptions import *
from grid2op._utils import extract_from_dict, save_to_dict
from grid2op.Space.GridObjects import GridObjects
from grid2op.Space.RandomObject import RandomObject


class SerializableSpace(GridObjects, RandomObject):
    """
    This class allows to serialize / de serialize the action space or observation space.

    It should not be used inside an Environment, as some functions of the action might not be compatible with
    the serialization, especially the checking of whether or not an BaseAction is legal or not.

    Attributes
    ----------

    subtype: ``type``
        Type use to build the template object :attr:`SerializableSpace.template_obj`. This type should derive
        from :class:`grid2op.BaseAction.BaseAction` or :class:`grid2op.BaseObservation.BaseObservation`.

    _template_obj: :class:`grid2op.GridObjects`
        An instance of the "*subtype*" provided used to provide higher level utilities, such as the size of the
        action (see :func:`grid2op.BaseAction.BaseAction.size`) or to sample a new BaseAction
        (see :func:`grid2op.BaseAction.BaseAction.sample`) for example.

    n: ``int``
        Size of the space

    shape: ``numpy.ndarray``, dtype:int
        Shape of each of the component of the Object if represented in a flat vector. An instance that derives from a
        GridObject (for example :class:`grid2op.BaseAction.BaseAction` or :class:`grid2op.BaseObservation.BaseObservation`) can be
        thought of as being concatenation of independant spaces. This vector gives the dimension of all the basic
        spaces they are made of.

    dtype: ``numpy.ndarray``, dtype:int
        Data type of each of the component of the Object if represented in a flat vector. An instance that derives from
        a GridObject (for example :class:`grid2op.BaseAction.BaseAction` or :class:`grid2op.BaseObservation.BaseObservation`) can be
        thought of as being concatenation of independant spaces. This vector gives the type of all the basic
        spaces they are made of.

    """
    def __init__(self, gridobj,
                 subtype=object):
        """

        subtype: ``type``
            Type of action used to build :attr:`SerializableActionSpace._template_act`. This type should derive
            from :class:`grid2op.BaseAction.BaseAction` or :class:`grid2op.BaseObservation.BaseObservation` .

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

        self._to_extract_vect = {}  # key: attr name, value: tuple: (beg_, end_, dtype)
        beg_ = 0
        end_ = 0
        for attr, size, dtype_ in zip(self._template_obj.attr_list_vect, self.shape, self.dtype):
            end_ += size
            self._to_extract_vect[attr] = (beg_, end_, dtype_)
            beg_ += size

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionnary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an BaseObservation Space (aka :class:`grid2op.BaseObservation.ObservartionHelper`)
            or the BaseAction Space (aka :class:`grid2op.BaseAction.ActionSpace`)
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
        Convert an action, represented as a vector to a valid :class:`BaseAction` instance

        Parameters
        ----------
        obj_as_vect: ``numpy.ndarray``
            A object living in a space represented as a vector (typically an :class:`grid2op.BaseAction.BaseAction` or an
            :class:`grid2op.BaseObservation.BaseObservation` represented as a numpy vector)

        Returns
        -------
        res: :class:`grid2op.Action.Action` or :class:`grid2op.Observation.Observation`
            The corresponding action (or observation) as an object (and not as a vector). The return type is given
            by the type of :attr:`SerializableSpace._template_obj`

        """
        res = copy.deepcopy(self._template_obj)
        res.from_vect(obj_as_vect)
        return res

    def extract_from_vect(self, obj_as_vect, attr_name):
        beg_, end_, dtype = self.get_indx_extract(attr_name)
        res = obj_as_vect[beg_:end_].astype(dtype)
        return res

    def get_indx_extract(self, attr_name):
        if attr_name not in self._to_extract_vect:
            raise Grid2OpException("Attribute \"{}\" is not found in the object of type \"{}\"."
                                   "".format(attr_name, self.subtype))
        res = self._to_extract_vect[attr_name]
        return res
