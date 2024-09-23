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

from grid2op.Exceptions import Grid2OpException
from grid2op.Space.space_utils import extract_from_dict, save_to_dict

from grid2op.Space.GridObjects import GridObjects
from grid2op.Space.RandomObject import RandomObject


class SerializableSpace(GridObjects, RandomObject):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        This is a higher level wrapper that allows to avoid code duplicates for
        the action_space and observation_space. None of the methods here should be
        used outside of `env.action_space` or `env.observation_space`

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

    def __init__(self, gridobj, subtype=object, _init_grid=True, _local_dir_cls=None):
        """

        subtype: ``type``
            Type of action used to build :attr:`SerializableActionSpace._template_act`. This type should derive
            from :class:`grid2op.BaseAction.BaseAction` or :class:`grid2op.BaseObservation.BaseObservation` .

        _init_grid: ``bool``
            Whether or not to call 'init_grid' in the subtype (to initialize the class). Do not modify unless you
            are certain of what you want to do
        """

        if not isinstance(subtype, type):
            raise Grid2OpException(
                'Parameter "subtype" used to build the Space should be a type (a class) and not an object '
                '(an instance of a class). It is currently "{}"'.format(type(subtype))
            )

        GridObjects.__init__(self)
        RandomObject.__init__(self)
        self._init_subtype = subtype  # do not use, use to save restore only !!!
        if _init_grid:
            self.subtype = subtype.init_grid(gridobj, _local_dir_cls=_local_dir_cls)
            from grid2op.Action import (
                BaseAction,
            )  # lazy loading to prevent circular reference

            if issubclass(self.subtype, BaseAction):
                # add the shunt data if needed by the action only
                self.subtype._add_shunt_data()
            # compute the class attribute "attr_list_set" from "attr_list_vect"
            self.subtype._update_value_set()
        else:
            self.subtype = subtype

        from grid2op.Action import BaseAction  # lazy import to avoid circular reference
        from grid2op.Observation import (
            BaseObservation,
        )  # lazy import to avoid circular reference

        if not issubclass(subtype, (BaseAction, BaseObservation)):
            raise RuntimeError(
                f'"subtype" should inherit either BaseAction or BaseObservation. Currently it '
                f'is "{subtype}"'
            )
        self._template_obj = self.subtype()
        self.n = self._template_obj.size()

        self.global_vars = None

        self._shape = self._template_obj.shapes()
        self._dtype = self._template_obj.dtypes()
        self.attr_list_vect = copy.deepcopy(type(self._template_obj).attr_list_vect)

        self._to_extract_vect = {}  # key: attr name, value: tuple: (beg_, end_, dtype)
        beg_ = 0
        end_ = 0
        for attr, size, dtype_ in zip(self.attr_list_vect, self.shape, self.dtype):
            end_ += size
            self._to_extract_vect[attr] = (beg_, end_, dtype_)
            beg_ += size

    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    def _custom_deepcopy_for_copy(self, new_obj):
        RandomObject._custom_deepcopy_for_copy(self, new_obj)

        # SerializableSpace
        new_obj._init_subtype = self._init_subtype  # const too
        new_obj.subtype = self.subtype
        new_obj._template_obj = self._template_obj.copy()
        new_obj.n = self.n
        new_obj.global_vars = copy.deepcopy(self.global_vars)
        new_obj._shape = copy.deepcopy(self._shape)
        new_obj._dtype = copy.deepcopy(self._dtype)
        new_obj.attr_list_vect = copy.deepcopy(self.attr_list_vect)  # TODO is this necessary, that's class attribute I think
        new_obj._to_extract_vect = copy.deepcopy(self._to_extract_vect)  # TODO is this necessary, that's class attribute I think

    @staticmethod
    def from_dict(dict_):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is used internally only to restore action_space or observation_space if they
            have been saved by `to_dict`. Do not
            attempt to use it in a different context.

        Allows the de-serialization of an object stored as a dictionary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an BaseObservation Space (aka :class:`grid2op.BaseObservation.ObservartionHelper`)
            or the BaseAction Space (aka :class:`grid2op.BaseAction.ActionSpace`)
            as a dictionary.

        Returns
        -------
        res: :class:`SerializableSpace`
            An instance of an SerializableSpace matching the dictionary.

        """

        if isinstance(dict_, str):
            path = dict_
            if not os.path.exists(path):
                raise Grid2OpException(
                    'Unable to find the file "{}" to load the grid2op classes'.format(
                        path
                    )
                )
            with open(path, "r", encoding="utf-8") as f:
                dict_ = json.load(fp=f)

        gridobj = GridObjects.from_dict(dict_)
        actionClass_str = extract_from_dict(dict_, "_init_subtype", str)
        actionClass_li = actionClass_str.split(".")
        _local_dir_cls = None  # TODO when reading back the data
        
        if actionClass_li[-1] in globals():
            subtype = globals()[actionClass_li[-1]]
        else:
            try:
                exec(
                    "from {} import {}".format(
                        ".".join(actionClass_li[:-1]), actionClass_li[-1]
                    )
                )
            except ModuleNotFoundError as exc_:
                # prior to grid2op 1.6.5 the Observation module was grid2op.Observation.completeObservation.CompleteObservation
                # after its grid2op.Observation.completeObservation.CompleteObservation
                # so I try here to make the python file lower case in order to import
                # the class correctly
                if len(actionClass_li) > 2:
                    test_str = actionClass_li[2]
                    actionClass_li[2] = test_str[0].lower() + test_str[1:]
                    exec(
                        "from {} import {}".format(
                            ".".join(actionClass_li[:-1]), actionClass_li[-1]
                        )
                    )
                else:
                    raise exc_

            # TODO make something better and recursive here
            try:
                subtype = eval(actionClass_li[-1])
            except NameError:
                if len(actionClass_li) > 1:
                    try:
                        subtype = eval(".".join(actionClass_li[1:]))
                    except Exception as exc_:
                        msg_err_ = (
                            'Impossible to find the module "{}" to load back the space (ERROR 1). '
                            'Try "from {} import {}"'
                        )
                        raise Grid2OpException(
                            msg_err_.format(
                                actionClass_str,
                                ".".join(actionClass_li[:-1]),
                                actionClass_li[-1],
                            )
                        )
                else:
                    msg_err_ = (
                        'Impossible to find the module "{}" to load back the space (ERROR 2). '
                        'Try "from {} import {}"'
                    )
                    raise Grid2OpException(
                        msg_err_.format(
                            actionClass_str,
                            ".".join(actionClass_li[:-1]),
                            actionClass_li[-1],
                        )
                    )
            except AttributeError:
                try:
                    subtype = eval(actionClass_li[-1])
                except Exception as exc_:
                    if len(actionClass_li) > 1:
                        msg_err_ = (
                            'Impossible to find the class named "{}" to load back the space (ERROR 3)'
                            "(module is found but not the class in it) Please import it via "
                            '"from {} import {}".'
                        )
                        msg_err_ = msg_err_.format(
                            actionClass_str,
                            ".".join(actionClass_li[:-1]),
                            actionClass_li[-1],
                        )
                    else:
                        msg_err_ = (
                            'Impossible to import the class named "{}" to load back the space (ERROR 4) '
                            "(the module is found but not the class in it)"
                        )
                        msg_err_ = msg_err_.format(actionClass_str)
                    raise Grid2OpException(msg_err_)
        # create the proper SerializableSpace class for this environment
        CLS = SerializableSpace.init_grid(gridobj, _local_dir_cls=_local_dir_cls)
        res = CLS(gridobj=gridobj, subtype=subtype, _init_grid=True, _local_dir_cls=_local_dir_cls)
        return res

    def cls_to_dict(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is used internally only to save action_space or observation_space for example. Do not
            attempt to use it in a different context.

        Serialize this object as a dictionary.

        Returns
        -------
        res: ``dict``
            A dictionary representing this object content. It can be loaded back with
            :func:`SerializableObservationSpace.cls_from_dict`

        """
        # TODO this is super weird that this is a regular method, but inherit from a class method !
        res = super().cls_to_dict()

        save_to_dict(
            res,
            self,
            "_init_subtype",
            lambda x: re.sub(
                "(<class ')|(\\.init_grid\\.<locals>\\.res)|('>)", "", "{}".format(x)
            ),
        )
        return res

    def size(self):
        """
        The size of any action converted to vector.

        Returns
        -------
        n: ``int``
            The size of the action space.

        Examples
        --------
        See :func:`GridObjects.size` for more information.

        """
        return self.n

    def from_vect(self, obj_as_vect, check_legit=True):
        """
        Convert an space (action space or observation space),
        represented as a vector to a valid :class:`BaseAction` instance. It works the
        same way for observations.

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

        Examples
        --------
        See :func:`GridObjects.from_vect` for more information.

        """
        res = copy.deepcopy(self._template_obj)
        res.from_vect(obj_as_vect, check_legit=check_legit)
        return res

    def extract_from_vect(self, obj_as_vect, attr_name):
        """
        This method allows you to extract only a given part of the observation / action  if this one
        is represented as a vector.

        Parameters
        ----------
        obj_as_vect: ``numpy.ndarray``
            the object (action or observation) represented as a vector.

        attr_name: ``str``
            the name of the attribute you want to extract from the object

        Returns
        -------
        res: ``numpy.ndarray``
            The value of the attribute with name `attr_name`

        Examples
        ---------
        We detail only the process for the observation, but it works the same way for the action too.

        .. code-block:: python

            import numpy as np
            import grid2op
            env = grid2op.make("l2rpn_case14_sandbox")

            # get the vector representation of an observation:
            obs = env.reset()
            obs_as_vect = obs.to_vect()

            # and now you can extract for example the load
            load_p = env.observation_space.extract_from_vect(obs_as_vect, "load_p")
            assert np.all(load_p == obs.load_p)
            # and this should assert to True

        """
        beg_, end_, dtype = self.get_indx_extract(attr_name)
        res = obj_as_vect[beg_:end_].astype(dtype)
        return res

    def get_indx_extract(self, attr_name):
        """
        Retrieve the type, the beginning and the end of a given attribute in the action or observation
        once it is represented as vector.

        [advanced usage] This is particularly useful to avoid parsing of all the observation / action when you want only
        to extract a subset of them (see example)

        Parameters
        ----------
        attr_name: ``str``
            The name of the attribute you want to extract information from

        Returns
        -------
        beg_: ``int``
            The first component of the vector that concerns the attribute
        end_: ``int``
            The las component of the vector that concerns the attribute
        dtype:
            The type of the attribute

        Examples
        --------
        This is an "advanced" function used to accelerate the study of an agent. Supposes you have an environment
        and you want to compute a runner from it. Then you want to have a quick look at the "relative flows" that
        this agent provides:

        .. code-block:: python

            import grid2op
            import os
            import numpy as np
            from grid2op.Runner import Runner
            from grid2op.Episode import EpisodeData

            ################
            # INTRO
            # create a runner
            env = grid2op.make("l2rpn_case14_sandbox")
            # see the documentation of the Runner if you want to change the agent.
            # in this case it will be "do nothing"
            runner = Runner(**env.get_params_for_runner())

            # execute it a given number of chronics
            nb_episode = 2
            path_save = "i_saved_the_runner_here"
            res = runner.run(nb_episode=nb_episode, path_save=path_save)

            # END INTRO
            ##################

            # now let's load only the flows for each of the computed episode
            li_episode = EpisodeData.list_episode(path_save)  # retrieve the list of where each episode is stored
            beg_, end_, dtype = env.observation_space.get_indx_extract("rho")
            observation_space_name = "observations.npz"

            for full_episode_path, episode_name in li_episode:
                all_obs = np.load(os.path.join(full_episode_path, observation_space_name))["data"]

                # and you use the function like this:
                all_flows = all_obs[:, beg_:end_].astype(dtype)

                # you can now do something with the computed flows
                # each row will be a time step, each column a powerline
                # you can have "nan" if the episode "game over" before the end.

        """
        if attr_name not in self._to_extract_vect:
            raise Grid2OpException(
                'Attribute "{}" is not found in the object of type "{}".'
                "".format(attr_name, self.subtype)
            )
        res = self._to_extract_vect[attr_name]
        return res
