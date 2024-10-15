# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

from grid2op.gym_compat import (GymEnv, GYM_AVAILABLE, GYMNASIUM_AVAILABLE)
import grid2op


CAN_TEST_ALL = True
if GYMNASIUM_AVAILABLE:
    from gymnasium.utils.env_checker import check_env
    from gymnasium.utils.env_checker import check_reset_return_type, check_reset_options
    try:
        from gymnasium.utils.env_checker import check_reset_seed
    except ImportError:
        # not present in most recent version of gymnasium, I copy pasted
        # it from an oldest version
        import gymnasium
        from logging import getLogger
        import inspect
        from copy import deepcopy
        import numpy as np
        logger = getLogger()
        
        
        def data_equivalence(data_1, data_2) -> bool:
            """Assert equality between data 1 and 2, i.e observations, actions, info.

            Args:
                data_1: data structure 1
                data_2: data structure 2

            Returns:
                If observation 1 and 2 are equivalent
            """
            if type(data_1) == type(data_2):
                if isinstance(data_1, dict):
                    return data_1.keys() == data_2.keys() and all(
                        data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
                    )
                elif isinstance(data_1, (tuple, list)):
                    return len(data_1) == len(data_2) and all(
                        data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
                    )
                elif isinstance(data_1, np.ndarray):
                    return data_1.shape == data_2.shape and np.allclose(
                        data_1, data_2, atol=0.00001
                    )
                else:
                    return data_1 == data_2
            else:
                return False
    
    
        def check_reset_seed(env: gymnasium.Env):
            """Check that the environment can be reset with a seed.

            Args:
                env: The environment to check

            Raises:
                AssertionError: The environment cannot be reset with a random seed,
                    even though `seed` or `kwargs` appear in the signature.
            """
            signature = inspect.signature(env.reset)
            if "seed" in signature.parameters or (
                "kwargs" in signature.parameters
                and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
            ):
                try:
                    obs_1, info = env.reset(seed=123)
                    assert (
                        obs_1 in env.observation_space
                    ), "The observation returned by `env.reset(seed=123)` is not within the observation space."
                    assert (
                        env.unwrapped._np_random  # pyright: ignore [reportPrivateUsage]
                        is not None
                    ), "Expects the random number generator to have been generated given a seed was passed to reset. Mostly likely the environment reset function does not call `super().reset(seed=seed)`."
                    seed_123_rng = deepcopy(
                        env.unwrapped._np_random  # pyright: ignore [reportPrivateUsage]
                    )

                    obs_2, info = env.reset(seed=123)
                    assert (
                        obs_2 in env.observation_space
                    ), "The observation returned by `env.reset(seed=123)` is not within the observation space."
                    if env.spec is not None and env.spec.nondeterministic is False:
                        assert data_equivalence(
                            obs_1, obs_2
                        ), "Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent."
                    assert (
                        env.unwrapped._np_random.bit_generator.state  # pyright: ignore [reportPrivateUsage]
                        == seed_123_rng.bit_generator.state
                    ), "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`."

                    obs_3, info = env.reset(seed=456)
                    assert (
                        obs_3 in env.observation_space
                    ), "The observation returned by `env.reset(seed=456)` is not within the observation space."
                    assert (
                        env.unwrapped._np_random.bit_generator.state  # pyright: ignore [reportPrivateUsage]
                        != seed_123_rng.bit_generator.state
                    ), "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`."

                except TypeError as e:
                    raise AssertionError(
                        "The environment cannot be reset with a random seed, even though `seed` or `kwargs` appear in the signature. "
                        f"This should never happen, please report this issue. The error was: {e}"
                    ) from e

                seed_param = signature.parameters.get("seed")
                # Check the default value is None
                if seed_param is not None and seed_param.default is not None:
                    logger.warning(
                        "The default seed argument in reset should be `None`, otherwise the environment will by default always be deterministic. "
                        f"Actual default: {seed_param.default}"
                    )
            else:
                raise gymnasium.error.Error(
                    "The `reset` method does not provide a `seed` or `**kwargs` keyword argument."
                )
                
                
elif GYM_AVAILABLE:
    from gym.utils.env_checker import check_env
    from gym.utils.env_checker import check_reset_return_type, check_reset_options, check_reset_seed
else:
    CAN_TEST_ALL = False


class Issue379Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
            self.gym_env = GymEnv(self.env)
    
    def tearDown(self) -> None:
        self.env.close()
        self.gym_env.close()
        return super().tearDown()
    
    def test_check_env(self):
        if CAN_TEST_ALL:
            check_reset_return_type(self.gym_env)
            check_reset_seed(self.gym_env)
            check_reset_options(self.gym_env)
        check_env(self.gym_env)
    

if __name__ == "__main__":
    unittest.main()
