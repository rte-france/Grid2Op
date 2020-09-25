# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numbers

from grid2op.Exceptions import *


def _get_default_aux(name, kwargs, defaultClassApp, _sentinel=None,
                     msg_error="Error when building the default parameter",
                     defaultinstance=None, defaultClass=None, build_kwargs={},
                     isclass=False):
    """
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
    err_msg = "Impossible to create the parameter \"{}\": "
    if _sentinel is not None:
        err_msg += "Impossible to get default parameters for building the environment. Please use keywords arguments."
        raise RuntimeError(err_msg)

    res = None
    # first seek for the parameter in the kwargs, and check it's valid
    if name in kwargs:
        res = kwargs[name]
        if isclass is False:
            # i must create an instance of a class. I check whether it's a instance.
            if not isinstance(res, defaultClassApp):
                if issubclass(defaultClassApp, numbers.Number):
                    try:
                        # if this is base numeric type, like float or anything, i try to convert to it (i want to
                        # accept that "int" are float for example.
                        res = defaultClassApp(res)
                    except:
                        # if there is any error, i raise the error message
                        raise EnvError(msg_error)
                else:
                    # if there is any error, i raise the error message
                    raise EnvError(msg_error)
        else:
            if not isinstance(res, type):
                raise EnvError("Parameter \"{}\" should be a type and not an instance. It means that you provided an "
                               "object instead of the class to build it.".format(name))
            # I must create a class, i check whether it's a subclass
            if not issubclass(res, defaultClassApp):
                raise EnvError(msg_error)

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
                    e.args = e.args + ("Cannot create and instance of {} with parameters \"{}\"".format(defaultClass, build_kwargs),)
                    raise
            elif defaultinstance is not None:
                if len(build_kwargs):
                    err_msg += "An instance is provided, yet kwargs to build it is also provided"
                    raise EnvError(err_msg.format(name))
                res = defaultinstance
            else:
                err_msg = " None of \"defaultClass\" and \"defaultinstance\" is provided."
                raise EnvError(err_msg.format(name))
        else:
            # I returning a class
            if len(build_kwargs):
                err_msg += "A class must be returned, yet kwargs to build it is also provided"
                raise EnvError(err_msg.format(name))
            if defaultinstance is not None:
                err_msg += "A class must be returned yet a default instance is provided"
                raise EnvError(err_msg.format(name))
            res = defaultClass

    return res
