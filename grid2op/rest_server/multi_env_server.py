# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
import os

import requests
import time
import grid2op
import numpy as np
from tqdm import tqdm
import asyncio
import aiohttp
import warnings
import subprocess
import sys

try:
    import ujson

    requests.models.complexjson = ujson
except ImportError as exc_:
    warnings.warn(
        "usjon is not installed. You could potentially get huge benefit if installing it"
    )

ERROR_NO_200 = "error due to not receiving 200 status"

NB_SUB_ENV = 4
ENV_NAME = "l2rpn_neurips_2020_track2_small"
SYNCH = True
NB_step = 300


PORTS = [3000 + i for i in range(NB_SUB_ENV)]  # TODO start them on the fly


class MultiEnvServer:
    def __init__(self, ports=PORTS, env_name=ENV_NAME, address="http://127.0.0.1"):
        warnings.warn(
            "This is an alpha feature and has absolutely not interest at the moment. Do not use unless "
            "you want to improve this feature yourself (-:"
        )
        self.my_procs = []
        for port in ports:
            p_ = subprocess.Popen(
                [
                    sys.executable,
                    "/home/benjamin/Documents/grid2op_dev/grid2op/rest_server/app.py",
                    "--port",
                    f"{port}",
                ],
                env=os.environ,
                # stdout=subprocess.DEVNULL,  # TODO logger
                # stderr=subprocess.DEVNULL  # TODO logger
            )
            self.my_procs.append(p_)

        self.nb_env = len(ports)
        self.ports = ports
        self.address = address
        self.li_urls = ["{}:{}".format(address, port) for port in ports]
        self.env_name = env_name
        self._local_env = grid2op.make(env_name)
        if SYNCH:
            self.session = requests.session()
        else:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.action_space = self._local_env.action_space
        self.observation_space = self._local_env.observation_space

        if SYNCH:
            answ_json = self._make_env_synch()
        else:
            answ_json = self.loop.run_until_complete(self._make_env_asynch())

        self.env_id = [int(el["id"]) for el in answ_json]
        self.obs = [el["obs"] for el in answ_json]

    def _make_env_synch(self):
        answs = []
        for url in self.li_urls:
            resp = self.session.get(f"{url}/make/{self.env_name}")
            answs.append(resp)
        import pdb

        pdb.set_trace()
        assert np.all(np.array([el.status_code for el in answs]) == 200), ERROR_NO_200
        answ_json = [el.json() for el in answs]
        return answ_json

    async def _make_env_asynch(self):
        answ_json = []
        async with aiohttp.ClientSession() as session:
            for url in self.li_urls:
                async with session.get(f"{url}/make/{self.env_name}") as resp:
                    if resp.status != 200:
                        raise RuntimeError(ERROR_NO_200)
                    answ_json.append(await resp.json())
        return answ_json

    def _step_synch(self, acts):
        answs = []
        for url, id_env, act in zip(self.li_urls, self.env_id, acts):
            resp = self.session.post(
                f"{url}/step/{self.env_name}/{id_env}", json={"action": act.to_json()}
            )
            answs.append(resp)
        answs = [el.json() for el in answs]
        return answs

    async def _step_asynch(self, acts):
        answs = []
        async with aiohttp.ClientSession() as session:
            for url, id_env, act in zip(self.li_urls, self.env_id, acts):
                async with session.post(
                    f"{url}/step/{self.env_name}/{id_env}",
                    json={"action": act.to_json()},
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(ERROR_NO_200)
                    answs.append(await resp.json())
        return answs

    def step(self, acts):
        if SYNCH:
            answ_json = self._step_synch(acts)
        else:
            answ_json = self.loop.run_until_complete(self._step_asynch(acts))

        obss = [el["obs"] for el in answ_json]
        rewards = [el["reward"] for el in answ_json]
        info = [el["info"] for el in answ_json]
        done = [el["done"] for el in answ_json]
        return obss, rewards, done, info

    def close(self):
        """close all the opened port"""
        for p_ in self.my_procs:
            p_.terminate()
            p_.kill()


if __name__ == "__main__":
    multi_env = MultiEnvServer()
    try:
        beg = time.perf_counter()
        for _ in tqdm(range(NB_step)):
            obs, reward, done, info = multi_env.step(
                [multi_env.action_space() for _ in range(multi_env.nb_env)]
            )
        end = time.perf_counter()
    finally:
        multi_env.close()
    print(
        f"Using {'synchronous' if SYNCH else 'asyncio'}, it took {end-beg:.2f}s to make {NB_step} steps "
        f"on {ENV_NAME} using {len(PORTS)} sub environment(s)."
    )
