# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions import Grid2OpException
from grid2op.Rules.BaseRules import BaseRules
from grid2op.Rules.AlwaysLegal import AlwaysLegal


class RulesChecker(object):
    """
    Class that define the rules of the game.

    """
    def __init__(self, legalActClass=AlwaysLegal):
        """

        Parameters
        ----------
        legalActClass: ``type``
            The class that will be used to tell if the actions are legal or not. The class must be given, and not
            an object of this class. It should derived from :class:`BaseRules`.
        """
        if not isinstance(legalActClass, type):
            raise Grid2OpException("Parameter \"legalActClass\" used to build the RulesChecker should be a "
                                   "type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))

        if not issubclass(legalActClass, BaseRules):
            raise Grid2OpException("Gamerules: legalActClass should be initialize with a class deriving "
                                   "from BaseRules and not {}".format(type(legalActClass)))
        self.legal_action = legalActClass()

    def __call__(self, action, env):
        """
        Says if an action is legal or not.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action that need to be tested

        env: :class:`grid2op.Environment.Environment`
            The current used environment.

        Returns
        -------
        is_legal: ``bool``
            Assess if the given action is legal or not. ``True``: the action is legal, ``False`` otherwise
        reason:
            A grid2op IllegalException given the reason for which the action is illegal
        """
        return self.legal_action(action, env)
