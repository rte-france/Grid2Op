# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from typing import Optional


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

    Notes
    -----

    In order to be reproducible, and to make proper use of the
    :func:`BaseAgent.seed` capabilities, you must absolutely NOT use the `random` python module (which will not
    be seeded) nor the `np.random` module and avoid any other "sources" of pseudo random numbers.

    You can adapt your code the following way. Instead of using `np.random` use `self.space_prng`.

    For example, if you wanted to write
    `np.random.randint(1,5)` replace it by `self.space_prng.randint(1,5)`. It is the same for `np.random.normal()`
    that is
    replaced by `self.space_prng.normal()`.

    You have an example of such usage in :func:`RandomAgent.my_act`.

    If you really need other sources of randomness (for example if you use tensorflow or torch) we strongly
    recommend you to overload the :func:`BaseAgent.seed` accordingly so that the neural networks are always initialized
    in the same order using the same weights.

    Examples
    ---------
    If you don't use any :class:`grid2op.Runner.Runner` we recommend using this method twice:

      1. to set the seed of the :class:`grid2op.Environment.Environment`
      2. to set the seed of your :class:`grid2op.Agent.BaseAgent`

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent # or any other agent of course. It might also be a custom you developed
        # create the environment
        env = grid2op.make()
        agent = RandomAgent(env.action_space)

        # and now set the seed
        env_seed = 42
        agent_seed = 12345
        env.seed(env_seed)
        agent.seed(agent_seed)

        # continue your experiments

    If you are using a :class:`grid2op.Runner.Runner` we recommend using the "env_seeds" and "agent_seeds" when
    calling the function :func:`grid2op.Runner.Runner.run` like this:

    .. code-block:: python

        import grid2op
        import numpy as np
        from grid2op.dtypes import dt_int
        from grid2op.Agent import RandomAgent # or any other agent of course. It might also be a custom you developed
        from grid2op.Runner import Runner

        np.random.seed(42)  # or any other seed of course :-)

        # create the environment
        env = grid2op.make()
        # NB setting a seed in this environment will have absolutely no effect on the runner

        # and now set the seed
        runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)

        # and now start your experiments
        nb_episode = 2
        maximum_int_poss = np.iinfo(dt_int).max  # this will be the maximum integer your computer can represent
        res = runner.run(nb_episode=nb_episode,
                         # generate the seeds for the agent
                         agent_seeds=[np.random.randint(0, maximum_int_poss) for _ in range(nb_episode)],
                         # generate the seeds for the environment
                         env_seeds=[np.random.randint(0, maximum_int_poss) for _ in range(nb_episode)]
                         )
        # NB for fully reproducible expriment you have to have called "np.random.seed" before using this method.

    """

    def __init__(self):
        self.space_prng : np.random.RandomState = np.random.RandomState()
        self.seed_used : Optional[int] = None

    def seed(self, seed):
        """
        INTERNAL

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            We do not recommend to use this function outside of the two examples given in the description of this class.

        Set the seed of the source of pseudo random number used for this RandomObject.

        Parameters
        ----------
        seed: ``int``
            The seed to be set.

        Returns
        -------
        res: ``tuple``
            The associated tuple of seeds used. Tuples are returned because in some cases, multiple objects are seeded
            with the same call to :func:`RandomObject.seed`

        """
        self.seed_used = seed
        if self.seed_used is not None:
            # in this case i have specific seed set. So i force the seed to be deterministic.
            self.space_prng.seed(seed=self.seed_used)
        return (self.seed_used,)

    def _custom_deepcopy_for_copy(self, new_obj):
        # RandomObject
        new_obj.space_prng = copy.deepcopy(self.space_prng)
        new_obj.seed_used = copy.deepcopy(self.seed_used)
