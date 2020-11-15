# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import requests
from tqdm import tqdm

from grid2op.MakeEnv import make

URL = " http://127.0.0.1:5000"
env_name = "l2rpn_case14_sandbox"
env_name = "l2rpn_neurips_2020_track2_small"

# create an environment
real_env = make(env_name)

print("Test \"make\"")
resp_make = requests.get(f"{URL}/make/{env_name}")

# check that the creation is working
if resp_make.status_code != 200:
    raise RuntimeError("Environment not created response not 200")

resp_make_json = resp_make.json()
if "id" not in resp_make_json:
    raise RuntimeError("Environment not created (due to id not in json)")
if "obs" not in resp_make_json:
    raise RuntimeError("Environment not created (due to obs not in json)")

real_obs = real_env.reset()
reic_obs_json = resp_make_json["obs"]
reic_obs = copy.deepcopy(real_obs)
reic_obs.set_game_over()
assert reic_obs != real_obs, "resetting the observation did not work"
reic_obs.from_json(reic_obs_json)
are_same = reic_obs == real_obs
diff_obs = reic_obs - real_obs
import pdb
pdb.set_trace()
assert are_same, "obs received and obs computed are not the same"

# make a step with do nothing
id_env = resp_make_json["id"]

print("Test \"step\"")
while True:
    act = real_env.action_space()
    obs, reward, done, info = real_env.step(act)
    resp_step = requests.post(f"{URL}/step/{env_name}/{id_env}", json={"action": act.to_json()})
    if resp_step.status_code != 200:
        raise RuntimeError("Step not successful not 200")
    resp_step_json = resp_step.json()
    if "obs" not in resp_step_json:
        raise RuntimeError("Environment not created (due to obs not in json)")

    reic_obs_json = resp_step_json["obs"]
    reic_obs_step = copy.deepcopy(real_obs)
    reic_obs_step.set_game_over()
    assert reic_obs_step != obs, "resetting the observation did not work"
    reic_obs_step.from_json(reic_obs_json)
    is_correct = reic_obs_step == obs
    assert is_correct, "obs received and obs computed are not the same after step"

    assert resp_step_json["done"] == done
    assert resp_step_json["reward"] == reward
    if done:
        break

print("Test \"perfs\"")
print("Time on a local env:")
env_perf = make(env_name)
obs = env_perf.reset()
with tqdm(desc="local env") as pbar:
    while True:
        act = real_env.action_space()
        obs, reward, done, info = env_perf.step(act)
        if done:
            break
        pbar.update(1)

print("Time on the remote env:")
resp_make_perf = requests.get(f"{URL}/make/{env_name}")
id_env = resp_make_perf.json()["id"]
obs = env_perf.reset()
with tqdm(desc="remote env") as pbar:
    while True:
        act = real_env.action_space()
        resp_step = requests.post(f"{URL}/step/{env_name}/{id_env}", json={"action": act.to_json()})
        resp_step_json = resp_step.json()
        reic_obs_json = resp_step_json["obs"]
        obs.from_json(reic_obs_json)
        if resp_step_json["done"]:
            break
        pbar.update(1)
