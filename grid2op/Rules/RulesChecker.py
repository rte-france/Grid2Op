# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings

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
        if isinstance(legalActClass, type):
            if not issubclass(legalActClass, BaseRules):
                raise Grid2OpException(
                    "Gamerules: legalActClass should be initialize with a class deriving "
                    "from BaseRules and not {}".format(type(legalActClass))
                )
            self.legal_action = legalActClass()
        else:
            if not isinstance(legalActClass, BaseRules):
                raise Grid2OpException(
                    'Parameter "legalActClass" used to build the Environment should be an instance of the '
                    'grid2op.BaseRules class, type provided is "{}"'.format(
                        type(legalActClass)
                    )
                )
            try:
                self.legal_action = copy.deepcopy(legalActClass)
            except Exception as exc_:
                warnings.warn("You passed the legal action as an instance that cannot be deepcopied. It will be "
                              "used 'as is', we do not garantee anything if you modify the original object.")
                self.legal_action = legalActClass

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
