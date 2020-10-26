# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Rules.BaseRules import BaseRules


class AlwaysLegal(BaseRules):
    """
    This subclass doesn't implement any rules regarding the legality of the actions. All actions are legal.

    """
    def __call__(self, action, env):
        """
        All actions being legal, this returns always true.
        See :func:`BaseRules.__call__` for a definition of the parameters of this function.

        """
        return True, None
