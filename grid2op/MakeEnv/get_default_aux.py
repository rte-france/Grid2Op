# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numbers
import copy
import warnings
from tarfile import ENCODING

from grid2op.Exceptions import *


def _get_default_aux(
    name,
    kwargs,
    defaultClassApp,
    _sentinel=None,
    msg_error="Error when building the default parameter",
    defaultinstance=None,
    defaultClass=None,
    build_kwargs={},
    isclass=False,
):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Helper to build default parameters forwarded to :class:`grid2op.Environment.Environment` for its creation.

    Exactly one of ``defaultinstance`` or ``defaultClass`` should be used, and set to not ``None``

    Parameters
    ----------
    name: ``str``
        Name of the argument to look for

    kwargs: ``dict``
        The key word arguments given to the :func:`make` function

    defaultClassApp; ``type``
        The default class to which the returned object should belong to. The final object should either be an instance
        of this ``defaultClassApp`` (if isclass is ``False``) or a subclass of this (if isclass is ``True``)

    _sentinel: ``None``
        Internal, do not use. Present to force key word arguments.

    msg_error: ``str`` or ``None``
        The message error to display if the object does not belong to ``defaultClassApp``

    defaultinstance: ``object`` or ``None``
        The default instance that will be returned. Note that if ``defaultinstance`` is not None, then
        ``defaultClass`` should be ``None`` and ``build_kwargs`` and empty dictionnary.

    defaultClass: ``type`` or ``None``
        The class used to build the default object. Note that if ``defaultClass`` is not None, then
        ``defaultinstance`` should be.

    build_kwargs:  ``dict``
        The keyword arguments used to build the final object (if ``isclass`` is ``True``). Note that:

          * if ``isclass`` is ``False``, this should be empty
          * if ``defaultinstance`` is not None, then this should be empty
          * This parameter should allow to create a valid object of type ``defaultClass``: it's key must be
            proper keys accepted by the class


    isclass: ``bool``
        Whether to build an instance of a class, or just return the class.


    Returns
    -------
    res:
        The parameters, either read from kwargs, or with its default value.

    """
    err_msg = 'Impossible to create the parameter "{}": '
    if _sentinel is not None:
        err_msg += "Impossible to get default parameters for building the environment. Please use keywords arguments."
        raise RuntimeError(err_msg)

    res = None
    # first seek for the parameter in the kwargs, and check it's valid
    if name in kwargs:
        res = kwargs[name]
        if defaultClassApp in (dict, list, set):# see https://github.com/rte-france/Grid2Op/issues/536
            try:
                res = copy.deepcopy(res)
            except copy.Error:
                warnings.warn(f"Impossible to copy mutable value for kwargs {name}. Make sure not to reuse "
                              f"the same kwargs for creating two environments."
                              "(more info on https://github.com/rte-france/Grid2Op/issues/536)")
        if isclass is None:
            # I don't know whether it's an object or a class
            error_msg_here = None
            res = None
            try:
                # I try to build it as an object
                res = _get_default_aux(
                    name,
                    kwargs=kwargs,
                    defaultClassApp=defaultClassApp,
                    _sentinel=_sentinel,
                    msg_error=msg_error,
                    defaultinstance=defaultinstance,
                    defaultClass=defaultClass,
                    build_kwargs=build_kwargs,
                    isclass=False,
                )
            except EnvError as exc1_:
                # I try to build it as a class
                try:
                    res = _get_default_aux(
                        name,
                        kwargs=kwargs,
                        defaultClassApp=defaultClassApp,
                        _sentinel=_sentinel,
                        msg_error=msg_error,
                        defaultinstance=defaultinstance,
                        defaultClass=defaultClass,
                        build_kwargs=build_kwargs,
                        isclass=True,
                    )
                except EnvError as exc2_:
                    # both fails !
                    error_msg_here = f"{exc1_} AND {exc2_}"

            if error_msg_here is not None:
                raise EnvError(error_msg_here)

        elif isclass is False:
            # i must create an instance of a class. I check whether it's a instance.
            if not isinstance(res, defaultClassApp):
                if issubclass(defaultClassApp, numbers.Number):
                    try:
                        # if this is base numeric type, like float or anything, i try to convert to it (i want to
                        # accept that "int" are float for example.
                        res = defaultClassApp(res)
                    except Exception as exc_:
                        # if there is any error, i raise the error message
                        raise EnvError(msg_error)
                else:
                    # if there is any error, i raise the error message
                    raise EnvError(msg_error)
        elif isclass is True:
            # so it should be a class
            if not isinstance(res, type):
                raise EnvError(
                    'Parameter "{}" should be a type and not an instance. It means that you provided an '
                    "object instead of the class to build it.".format(name)
                )
            # I must create a class, i check whether it's a subclass
            if not issubclass(res, defaultClassApp):
                raise EnvError(msg_error)
        else:
            raise EnvError(
                'Impossible to use the "_get_default_aux" function with "isclass" kwargs being different '
                "from None, True and False"
            )

    if res is None:
        # build the default parameter if not found

        if isclass is False:
            # i need building an instance
            if defaultClass is not None:
                if defaultinstance is not None:
                    err_msg += "Impossible to build an environment with both a default instance, and a default class"
                    raise EnvError(err_msg.format(name))
                try:
                    res = defaultClass(**build_kwargs)
                except Exception as e:
                    e.args = e.args + (
                        'Cannot create and instance of {} with parameters "{}"'.format(
                            defaultClass, build_kwargs
                        ),
                    )
                    raise
            elif defaultinstance is not None:
                if len(build_kwargs):
                    err_msg += "An instance is provided, yet kwargs to build it is also provided"
                    raise EnvError(err_msg.format(name))
                res = defaultinstance
            else:
                err_msg = ' None of "defaultClass" and "defaultinstance" is provided.'
                raise EnvError(err_msg.format(name))
        else:
            # I returning a class
            if len(build_kwargs):
                err_msg += (
                    "A class must be returned, yet kwargs to build it is also provided"
                )
                raise EnvError(err_msg.format(name))
            if defaultinstance is not None:
                err_msg += "A class must be returned yet a default instance is provided"
                raise EnvError(err_msg.format(name))
            res = defaultClass
    return res
