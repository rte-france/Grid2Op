# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.dtypes import dt_int, dt_float
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Observation.completeObservation import CompleteObservation


class NoisyObservation(BaseObservation):
    """
    This class represent a complete observation (in the sens that all attributes
    of an :attr:`CompleteObservation` are accessible) but some of them are
    "noisy".

    That is, the observation that the agent has access to is not
    exactly the same as the environment internal values.

    The affected attributes are :

    - load_p: \*= lognormal (to keep the sign)
    - load_q: \*= lognormal (to keep the sign)
    - gen_p: \*= lognormal (to keep the sign)
    - gen_q: \*= lognormal (to keep the sign)
    - p_or += normal
    - p_ex += normal
    - q_or += normal
    - q_ex += normal
    - a_or: \*= lognormal (to keep the sign)
    - a_ex: \*= lognormal (to keep the sign)
    - rho: same noise as a_or (because rho is not "physical" it's the result of a computation)
    - storage_power += normal

    It can be used to emuate the acquisition of data coming from noisy sensors for
    example.

    Examples
    --------

    It can be used as follow:

    .. code-block:: python

        import grid2op

        env_name = "l2rpn_case14_sandbox"  # or any other name
        kwargs_observation = {"sigma_load_p": 0.1, "sigma_gen_p": 1.0}  # noise of the observation
        env = grid2op.make(env_name,
                           observation_class=NoisyObservation,
                           kwargs_observation=kwargs_observation)

        # do whatever you want with env !

    """

    attr_list_vect = CompleteObservation.attr_list_vect
    attr_list_json = CompleteObservation.attr_list_json
    attr_list_set = CompleteObservation.attr_list_set

    def __init__(
        self,
        obs_env=None,
        action_helper=None,
        random_prng=None,
        kwargs_env=None,
        sigma_load_p=0.01,  # multiplicative (log normal)
        sigma_load_q=0.01,  # multiplicative (log normal)
        sigma_gen_p=0.01,  # multiplicative (log normal)
        sigma_gen_q=0.01,  # multiplicative (log normal)
        sigma_a=0.01,  # multiplicative (log normal) same for a_or and a_ex
        sigma_p=0.1,  # additive (normal) same for p_or and p_ex
        sigma_q=0.1,  # additive (normal) same for q_or and q_ex
        sigma_storage=0.1,  # additive (normal)
    ):

        BaseObservation.__init__(
            self,
            obs_env=obs_env,
            action_helper=action_helper,
            random_prng=random_prng,
            kwargs_env=kwargs_env
        )
        self._dictionnarized = None
        self._sigma_load_p = sigma_load_p  # multiplicative (log normal)
        self._sigma_load_q = sigma_load_q  # multiplicative (log normal)
        self._sigma_gen_p = sigma_gen_p  # multiplicative (log normal)
        self._sigma_gen_q = sigma_gen_q  # multiplicative (log normal)
        self._sigma_a = sigma_a  # multiplicative (log normal) same for a_or and a_ex

        self._sigma_p = sigma_p  # additive (normal) same for p_or and p_ex
        self._sigma_q = sigma_q  # additive (normal) same for q_or and q_ex
        self._sigma_storage = sigma_storage  # additive (normal)

    def update(self, env, with_forecast=True):
        # reset the matrices
        self._reset_matrices()
        self.reset()

        # update as if the data were complete
        self._update_obs_complete(env, with_forecast=with_forecast)

        # multiplicative noise
        mult_load_p = self.random_prng.lognormal(
            mean=0.0, sigma=self._sigma_load_p, size=self.load_p.shape
        )
        self.load_p[:] *= mult_load_p
        mult_load_q = self.random_prng.lognormal(
            mean=0.0, sigma=self._sigma_load_q, size=self.load_p.shape
        )
        self.load_q[:] *= mult_load_q
        mult_gen_p = self.random_prng.lognormal(
            mean=0.0, sigma=self._sigma_gen_p, size=self.gen_p.shape
        )
        self.gen_p[:] *= mult_gen_p
        mult_gen_q = self.random_prng.lognormal(
            mean=0.0, sigma=self._sigma_gen_q, size=self.gen_q.shape
        )
        self.gen_q[:] *= mult_gen_q
        mult_aor = self.random_prng.lognormal(
            mean=0.0, sigma=self._sigma_a, size=self.a_or.shape
        )
        self.a_or[:] *= mult_aor
        self.rho[:] *= mult_aor
        mult_a_ex = self.random_prng.lognormal(
            mean=0.0, sigma=self._sigma_a, size=self.a_ex.shape
        )
        self.a_ex[:] *= mult_a_ex

        # additive noise
        add_por = self.random_prng.normal(
            loc=0.0,
            scale=self._sigma_p,  # 0.01 * np.abs(self.p_or),
            size=self.p_or.shape,
        )
        self.p_or[:] += add_por

        add_pex = self.random_prng.normal(
            loc=0.0,
            scale=self._sigma_p,  # 0.01 * np.abs(self.p_ex),
            size=self.p_or.shape,
        )
        self.p_ex[:] += add_pex

        add_qor = self.random_prng.normal(
            loc=0.0,
            scale=self._sigma_q,  # 0.01 * np.abs(self.q_or),
            size=self.p_or.shape,
        )
        self.q_or[:] += add_qor

        add_qex = self.random_prng.normal(
            loc=0.0,
            scale=self._sigma_q,  # 0.01 * np.abs(self.q_ex),
            size=self.p_or.shape,
        )
        self.q_ex[:] += add_qex

        add_storp = self.random_prng.normal(
            loc=0.0,
            scale=self._sigma_storage,  # 0.01 * np.abs(self.storage_power),
            size=self.storage_power.shape,
        )
        self.storage_power[:] += add_storp
