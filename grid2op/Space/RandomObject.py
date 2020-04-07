# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np


class RandomObject(object):
    """
    Utility class to deal with randomness in some aspect of the game (chronics, action_space, observation_space for
    examples.

    Attributes
    ----------
    space_prng: ``numpy.random.RandomState``
        The random state of the observation (in case of non deterministic observations or BaseAction.
        This should not be used at the
        moment)

    seed_used: ``int``
        The seed used throughout the episode in case of non deterministic observations or action.

    """
    def __init__(self):
        self.space_prng = np.random.RandomState()
        self.seed_used = None

    def seed(self, seed):
        """
        Use to set the seed in case of non deterministic observations.
        :param seed:
        :return:
        """
        self.seed_used = seed
        if self.seed_used is not None:
            # in this case i have specific seed set. So i force the seed to be deterministic.
            self.space_prng.seed(seed=self.seed_used)

