# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from grid2op.Reward import BaseReward
from grid2op.Action._BackendAction import _BackendAction


class N1Reward(BaseReward):
    """
    This class implements the "n-1" reward, which returns the maximum flows after a powerline

    Examples
    --------

    This can be used as:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import N1Reward
        L_ID = 0
        env = grid2op.make("l2rpn_case14_sandbox",
                    reward_class=N1Reward(l_id=L_ID)
                    )
        obs = env.reset()
        obs, reward, *_ = env.step(env.action_space())
        print(f"reward: {reward:.3f}")
        print("We can check that it is exactly like 'simulate' on the current step the disconnection of the same powerline")
        obs_n1, *_ = obs.simulate(env.action_space({"set_line_status": [(L_ID, -1)]}), time_step=0)
        print(f"\tmax flow after disconnection of line {L_ID}: {obs_n1.rho.max():.3f}")

    Notes
    -----
    It is also possible to use the `other_rewards` argument to simulate multiple powerline disconnections, for example:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import N1Reward
        L_ID = 0
        env = grid2op.make("l2rpn_case14_sandbox",
                           other_rewards={f"line_{l_id}": N1Reward(l_id=l_id)  for l_id in [0, 1]}
                           )
        obs = env.reset()
        obs, reward, *_ = env.step(env.action_space())
        print(f"reward: {reward:.3f}")
        print("We can check that it is exactly like 'simulate' on the current step the disconnection of the same powerline")
        obs_n1, *_ = obs.simulate(env.action_space({"set_line_status": [(L_ID, -1)]}), time_step=0)
        print(f"\tmax flow after disconnection of line {L_ID}: {obs_n1.rho.max():.3f}")

    """

    def __init__(self, l_id=0, logger=None):
        BaseReward.__init__(self, logger=logger)
        self._backend = None
        self._backend_action = None
        self.l_id = l_id

    def initialize(self, env):
        self._backend = env.backend.copy()
        bk_act_cls = _BackendAction.init_grid(env.backend)
        self._backend_action = bk_act_cls()

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return self.reward_min
        self._backend_action.reset()
        act = env.backend.get_action_to_set()
        th_lim = env.get_thermal_limit()
        th_lim[th_lim <= 1] = 1  # assign 1 for the thermal limit
        this_n1 = copy.deepcopy(act)
        self._backend_action += this_n1
            
        self._backend.apply_action(self._backend_action)
        self._backend._disconnect_line(self.l_id)
        div_exc_ = None
        try:
            # TODO there is a bug in lightsimbackend that make it crash instead of diverging
            conv, div_exc_ = self._backend.runpf()
        except Exception as exc_:
            conv = False
            div_exc_ = exc_

        if conv:
            flow = self._backend.get_line_flow()
            res = (flow / th_lim).max()
        else:
            self.logger.info(f"Divergence of the backend at step {env.nb_time_step} for N1Reward with error `{div_exc_}`")
            res = -1
        return res

    def close(self):
        self._backend.close()
        del self._backend
        self._backend = None
