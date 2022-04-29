# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings
import requests

try:
    import ujson

    requests.models.complexjson = ujson
except ImportError as exc_:
    warnings.warn(
        "usjon is not installed. You could potentially get huge benefit if installing it"
    )

import time
import numpy as np
from tqdm import tqdm

from grid2op.MakeEnv import make

try:
    from lightsim2grid import LightSimBackend

    bkclass = LightSimBackend
    # raise ImportError()
except ImportError as exc_:
    from grid2op.Backend import PandaPowerBackend

    bkclass = PandaPowerBackend
    warnings.warn(
        "lightsim2grid is not installed. You could potentially get huge benefit if installing it"
    )
    pass

URL = " http://127.0.0.1:5000"
env_name = "l2rpn_case14_sandbox"
env_name = "l2rpn_neurips_2020_track1_small"
env_name = "l2rpn_neurips_2020_track2_small"


def assert_rec_equal(li1, li2):
    try:
        len(li1)
        assert len(li1) == len(li2), "wrong seed length"
        for (el1, el2) in zip(li1, li2):
            assert_rec_equal(el1, el2)
    except Exception as exc_:
        if isinstance(li1, np.ndarray):
            if li1.shape == ():
                li1 = []
        assert li1 == li2, "wrong seed value"


if __name__ == "__main__":
    # create the real environment
    real_env = make(env_name, backend=bkclass())
    real_obs = real_env.reset()
    client = requests.session()

    print('Test "make"')
    # test the "make" endpoint of the API
    resp_make = client.get(f"{URL}/make/{env_name}")

    # check that the creation is working
    if resp_make.status_code != 200:
        raise RuntimeError("Environment not created response not 200")
    resp_make_json = resp_make.json()
    if "id" not in resp_make_json:
        raise RuntimeError("Environment not created (due to id not in json)")
    if "obs" not in resp_make_json:
        raise RuntimeError("Environment not created (due to obs not in json)")

    if env_name == "l2rpn_case14_sandbox":
        # the other envs are stochastic so this test cannot work right now (this is why we used the "seed" just after)
        reic_obs_json = resp_make_json["obs"]
        reic_obs = copy.deepcopy(real_obs)
        reic_obs.set_game_over()
        assert reic_obs != real_obs, "resetting the observation did not work"
        reic_obs.from_json(reic_obs_json)
        are_same = reic_obs == real_obs
        diff_obs = reic_obs - real_obs
        assert are_same, "obs received and obs computed are not the same"

    # make a step with do nothing
    id_env = resp_make_json["id"]

    print('Test "seed"')
    seed_used = 0
    resp_seed = client.post(
        f"{URL}/seed/{env_name}/{id_env}",
        json={"seed": seed_used},
        # headers={"X-CSRFToken": csrf_token}
    )
    if resp_seed.status_code != 200:
        raise RuntimeError("Environment not seeded response not 200")
    resp_seed_json = resp_seed.json()
    if "seeds" not in resp_seed_json:
        raise RuntimeError("Environment not seeded (due to seeds not in json)")
    res_seed = real_env.seed(seed_used)
    res_seed = list(res_seed)
    assert_rec_equal(res_seed, resp_seed_json["seeds"])

    print('Test "reset"')
    resp_reset = client.get(f"{URL}/reset/{env_name}/{id_env}")
    if resp_reset.status_code != 200:
        raise RuntimeError("Environment not reset response not 200")
    resp_reset_json = resp_reset.json()
    if "obs" not in resp_reset_json:
        raise RuntimeError("Environment not reset (due to obs not in json)")
    real_obs = real_env.reset()
    reic_obs_json = resp_reset_json["obs"]
    reic_obs = copy.deepcopy(real_obs)
    reic_obs.set_game_over()
    assert reic_obs != real_obs, "resetting the observation did not work"
    reic_obs.from_json(reic_obs_json)
    are_same = reic_obs == real_obs
    obs_diff, attr_diff = reic_obs.where_different(real_obs)
    for el in attr_diff:
        if np.max(np.abs(getattr(obs_diff, el))) > 1e-4:
            tmp_ = np.max(np.abs(getattr(obs_diff, el)))
            import pdb

            pdb.set_trace()
            raise RuntimeError(
                f"ERROR: after reset, attribute {el} is not the same (max={tmp_:.6f})"
            )
    if not are_same:
        warnings.warn(
            "obs received and obs computed are not the exactly the same "
            "(but equal up to some small value (1e-4))"
        )

    print('Test "set_id"')
    resp_set_id = client.post(f"{URL}/set_id/{env_name}/{id_env}", json={"id": 0})
    if resp_set_id.status_code != 200:
        raise RuntimeError("set_id not working: response is not 200")
    resp_set_id_json = resp_set_id.json()
    if "info" not in resp_set_id_json:
        raise RuntimeError("set_id not working: info is not in response")

    resp_reset = client.get(f"{URL}/reset/{env_name}/{id_env}")
    if resp_seed.status_code != 200:
        raise RuntimeError("Environment not reset response not 200")
    resp_reset_json = resp_reset.json()
    if "obs" not in resp_reset_json:
        raise RuntimeError("Environment not reset (due to obs not in json)")
    real_env.set_id(0)
    real_obs = real_env.reset()
    reic_obs_json = resp_reset_json["obs"]
    reic_obs = copy.deepcopy(real_obs)
    reic_obs.set_game_over()
    assert reic_obs != real_obs, "resetting the observation did not work"
    reic_obs.from_json(reic_obs_json)
    are_same = reic_obs == real_obs
    assert are_same, "obs received and obs computed are not the same"

    print('Test "set_thermal_limit"')
    th_lim = real_env.get_thermal_limit().tolist()
    resp_set_thermal_limit = client.post(
        f"{URL}/set_thermal_limit/{env_name}/{id_env}", json={"thermal_limits": th_lim}
    )
    if resp_set_thermal_limit.status_code != 200:
        raise RuntimeError("set_thermal_limit not working: response is not 200")
    resp_set_thermal_limit_json = resp_set_thermal_limit.json()
    if "env_name" not in resp_set_thermal_limit_json:
        raise RuntimeError("set_thermal_limit not working: info is not in response")

    print('Test "get_thermal_limit"')
    resp_get_thermal_limit = client.get(f"{URL}/get_thermal_limit/{env_name}/{id_env}")
    if resp_get_thermal_limit.status_code != 200:
        raise RuntimeError("get_thermal_limit not working: response is not 200")
    resp_get_thermal_limit_json = resp_get_thermal_limit.json()
    if "thermal_limit" not in resp_get_thermal_limit_json:
        raise RuntimeError(
            "get_thermal_limit not working: thermal_limit is not in response"
        )
    assert (
        resp_get_thermal_limit_json["thermal_limit"] == th_lim
    ), "get_thermal_limit not working: wrong thermal limit"

    print('Test "get_path_env"')
    resp_get_path_env = client.get(f"{URL}/get_path_env/{env_name}/{id_env}")
    if resp_get_path_env.status_code != 200:
        raise RuntimeError("get_path_env not working: response is not 200")
    resp_get_path_env_json = resp_get_path_env.json()
    if "path" not in resp_get_path_env_json:
        raise RuntimeError("get_path_env not working: path is not in response")
    assert (
        resp_get_path_env_json["path"] == real_env.get_path_env()
    ), "get_path_env not working: wrong path"

    print('Test "fast_forward_chronics"')
    nb_step_forward = 10
    resp_fast_forward_chronics = client.post(
        f"{URL}/fast_forward_chronics/{env_name}/{id_env}",
        json={"nb_step": nb_step_forward},
    )
    if resp_fast_forward_chronics.status_code != 200:
        raise RuntimeError("set_id not working: response is not 200")
    resp_fast_forward_chronics_json = resp_fast_forward_chronics.json()
    if "env_name" not in resp_fast_forward_chronics_json:
        raise RuntimeError(
            "get_thermal_limit not working: thermal_limit is not in response"
        )
    act = real_env.action_space()
    real_env.fast_forward_chronics(nb_step_forward)
    obs, reward, done, info = real_env.step(act)
    resp_step = client.post(
        f"{URL}/step/{env_name}/{id_env}", json={"action": act.to_json()}
    )
    # check obs are equals
    reic_obs_json = resp_step.json()["obs"]
    reic_obs = copy.deepcopy(obs)
    reic_obs.set_game_over()
    assert reic_obs != real_obs, "resetting the observation did not work"
    reic_obs.from_json(reic_obs_json)
    are_same = reic_obs == obs
    assert (
        are_same
    ), "obs received and obs computed are not the same after fast forwarding"

    print('Test "step"')
    real_obs = real_env.reset()
    resp_reset = client.get(f"{URL}/reset/{env_name}/{id_env}")
    if resp_seed.status_code != 200:
        raise RuntimeError(
            "Environment not reset response not 200, fail just before assessing step"
        )

    nb_step = 0
    obs_me = copy.deepcopy(real_obs)
    while True:
        act = real_env.action_space()
        obs, reward, done, info = real_env.step(act)
        resp_step = client.post(
            f"{URL}/step/{env_name}/{id_env}", json={"action": act.to_json()}
        )
        if resp_step.status_code != 200:
            raise RuntimeError("Step not successful not 200")
        resp_step_json = resp_step.json()
        if "obs" not in resp_step_json:
            raise RuntimeError("Environment not created (due to obs not in json)")

        assert resp_step_json["done"] == done
        assert resp_step_json["reward"] == reward
        if done:
            break

        reic_obs_json = resp_step_json["obs"]
        obs_me.set_game_over()
        assert (
            obs_me != obs
        ), f"resetting the observation did not work for step {nb_step}"
        obs_me.from_json(reic_obs_json)
        is_correct = obs_me == obs
        try:
            assert (
                is_correct
            ), f"obs received and obs computed are not the same after step for step {nb_step}"
        except AssertionError as exc_:
            obs_diff, attr_diff = obs_me.where_different(obs)
            # import pdb
            # pdb.set_trace()
        nb_step += 1

    print('Test "close"')
    resp_close = client.get(f"{URL}/close/{env_name}/{id_env}")
    if resp_close.status_code != 200:
        raise RuntimeError("close not working: response is not 200")
    resp_close_json = resp_close.json()
    if "env_name" not in resp_close_json:
        raise RuntimeError("close not working: env_name is not in response")
    # TODO test all methods fails (get_thermal_limit apparently work...)

    print('Test "perfs"')
    print(f"Time on a local env: (using {bkclass.__name__})")
    env_perf = make(env_name, backend=bkclass())
    env_perf.reset()
    env_perf.seed(seed_used)
    obs = env_perf.reset()
    time_for_step = 0
    nb_step_local = 0
    with tqdm(desc="local env") as pbar:
        while True:
            act = real_env.action_space()
            beg_step = time.perf_counter()
            obs, reward, done, info = env_perf.step(act)
            time_for_step += time.perf_counter() - beg_step
            if done:
                break
            nb_step_local += 1
            pbar.update(1)

    print("Time on the remote env:")
    resp_make_perf = client.get(f"{URL}/make/{env_name}")
    id_env_perf = resp_make_perf.json()["id"]
    _ = client.post(f"{URL}/seed/{env_name}/{id_env_perf}", json={"seed": seed_used})
    _ = client.get(f"{URL}/reset/{env_name}/{id_env_perf}")
    time_for_step_api = 0.0
    time_for_all_api = 0.0
    time_convert = 0.0
    time_get_json = 0.0
    nb_step_api = 0
    with tqdm(desc="remote env") as pbar:
        while True:
            act = real_env.action_space()
            beg_step = time.perf_counter()
            act_as_json = act.to_json()
            resp_step = client.post(
                f"{URL}/step/{env_name}/{id_env_perf}", json={"action": act_as_json}
            )
            after_step = time.perf_counter()
            time_for_step_api += after_step - beg_step
            resp_step_json = resp_step.json()
            time_get_json += time.perf_counter() - after_step
            reic_obs_json = resp_step_json["obs"]
            beg_convert = time.perf_counter()
            obs.from_json(reic_obs_json)
            time_convert += time.perf_counter() - beg_convert
            time_for_all_api += time.perf_counter() - beg_step
            if resp_step_json["done"]:
                break
            pbar.update(1)
            nb_step_api += 1

    print(f"\tEnv name: {env_name} with {real_env.n_sub} substations")
    print(f"\tNumber of step for local env {nb_step_local}")
    print(f"\tNumber of step for api env {nb_step_api}")
    print(f"\tTime to compute all, for the normal env: {time_for_step:.2f}s")
    print(f"\tTime to compute all, for the api env: {time_for_all_api:.2f}s")
    print(f"\t\tTime to do the step, for the api env: {time_for_step_api:.2f}s")
    print(f"\t\tTime to get the json from the http request: {time_get_json:.2f}s")
    print(f"\t\tTime to convert from json: {time_convert:.2f}s")
    print(f"\tSpeed up (for normal env): {time_for_all_api/time_for_step:.2f}")
