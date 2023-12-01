# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions import Grid2OpException


class DomainException(Grid2OpException):
    """This class represents the exceptions that occur in the
    _verify_domains method of MultiAgentEnv. It occurs when 
    the given domain by the user is not valid.
    """
    pass


class MultiAgentStillBeta(UserWarning):
    """
    Multi agent is currently a beta feature. API is subject to change
    """
    pass


class MissingFeature(MultiAgentStillBeta):
    """This feature is currently missing, if you need it, either implement it
    and make a pull request. Or make a feature request at :\n
    https://github.com/rte-france/Grid2Op/issues/new?assignees=&labels=enhancement,multi-agents&template=feature_request.md&title=[Missing Feature]:
    """
