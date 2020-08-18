# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Rules.LookParam import LookParam
from grid2op.Rules.PreventReconnection import PreventReconnection


class DefaultRules(LookParam, PreventReconnection):
    """
    This subclass combine both :class:`LookParam` and :class:`PreventReconnection`.
    An action is declared legal if and only if:

      - It doesn't disconnect / reconnect more power lines than  what stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to act on more substations that what is stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to reconnect a powerline out of service.

    """
    def __call__(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the _parameters of this function.
        """
        is_legal, reason = LookParam.__call__(self, action, env)
        if not is_legal:
            return False, reason

        return PreventReconnection.__call__(self, action, env)
