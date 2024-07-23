# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions.grid2OpException import Grid2OpException


# ambiguous action
class AmbiguousAction(Grid2OpException):
    """
    This exception indicate that the :class:`grid2op.BaseAction` is ambiguous.
    It could be understood differently according
    to the backend used.

    Such a kind of action are forbidden in this package. These kind of exception are mainly thrown by the
    :class:`grid2op.BaseAction.BaseAction` in
    the :func:`grid2op.BaseAction.update` and :func:`grid2op.BaseAction.__call__` methods.

    As opposed to a :class:`IllegalAction` an :class:`AmbiguousAction` is forbidden for all the backend,
    in all the scenarios.

    It doesn't depend on the implemented rules.
    """

    pass


class InvalidLineStatus(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the
     :class:`grid2op.BaseAction.BaseAction` is ambiguous due to powerlines manipulation.
    """

    pass


class InvalidStorage(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the
     :class:`grid2op.BaseAction.BaseAction` is ambiguous due to storage unit manipulation.
    """

    pass


class UnrecognizedAction(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that the
    :class:`grid2op.BaseAction.BaseAction` is  ambiguous due to the bad formatting of the action.
    """

    pass


class InvalidNumberOfLoads(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction` is ambiguous because an incorrect number of loads tries to be modified.
    """

    pass


class InvalidNumberOfGenerators(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    is ambiguous because an incorrect number of generator tries to be modified.
    """

    pass


class InvalidNumberOfLines(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    is ambiguous because an incorrect number of lines tries to be modified.
    """

    pass


class InvalidNumberOfObjectEnds(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    is ambiguous because an incorrect number of object at a substation try to be modified.
    """

    pass


class InvalidBusStatus(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    try to both "set" and "switch" some bus to which an object is connected.
    """

    pass


class InvalidRedispatching(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    try to apply an invalid redispatching strategy.
    """

    pass

class InvalidFlexibility(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    try to apply an invalid Flexibility strategy.
    """

    pass

class InvalidCurtailment(AmbiguousAction):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that
    the :class:`grid2op.BaseAction.BaseAction`
    try to apply an invalid curtailment strategy.
    """

    pass


class GeneratorTurnedOnTooSoon(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that a generator has been turned on
    before gen_min_up_time time steps.
    """

    pass


class GeneratorTurnedOffTooSoon(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that a generator has been turned off
    before gen_min_down_time time steps.
    """

    pass


class NotEnoughGenerators(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that there is not enough turned off
    generators to meet the demand.
    """

    pass


class NonFiniteElement(InvalidRedispatching):
    """
    This is a more precise exception than :class:`AmbiguousAction` indicating that an action / observation
    non initialized (full of Nan)
    has been loaded by the "from_vect" method.
    """

    pass

class AmbiguousActionRaiseAlert(AmbiguousAction):
    """Raise if the type of action is ambiguous due to the 'raiseAlert' part"""
    pass