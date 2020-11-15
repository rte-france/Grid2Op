# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from flask import Flask
from flask import abort
from flask import make_response, jsonify
from flask import request

from grid2op.MakeEnv import make

app = Flask(__name__)

ALL_ENV = {}


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/make/<env_name>')
def make_env(env_name):
    try:
        env = make(env_name)
    except Exception as exc_:
        return make_response(jsonify({'error': f"Impossible to create environment \"{env_name}\" with error:\n"
                                               f"{exc_}"}), 400)

    if env_name not in ALL_ENV:
        # create an environment with that name
        ALL_ENV[env_name] = [env]
    else:
        ALL_ENV[env_name].append(env)

    resp = {"id": len(ALL_ENV[env_name]) - 1}
    obs = env.reset()
    # TODO have a "to list"
    resp["obs"] = obs.to_json()
    return jsonify(resp)


def to_correct_json(info):
    # TODO
    res = {}
    res["disc_lines"] = [int(el) for el in info["disc_lines"]]
    res["is_illegal"] = bool(info["is_illegal"])
    res["is_ambiguous"] = bool(info["is_ambiguous"])
    res["is_dispatching_illegal"] = bool(info["is_dispatching_illegal"])
    res["is_illegal_reco"] = bool(info["is_illegal_reco"])
    if info["opponent_attack_line"] is not None:
        res["opponent_attack_line"] = [bool(el) for el in info["opponent_attack_line"]]
    else:
        res["opponent_attack_line"] = None
    res["exception"] = [f"{exc_}" for exc_ in info["exception"]]
    return res


@app.route('/step/<env_name>/<env_id>', methods=["POST"])
def step(env_name, env_id):
    # for now just run a do nothing
    if env_name not in ALL_ENV:
        return make_response(jsonify({'error': f"environment \"{env_name}\" does not exists"}), 400)

    li_env = ALL_ENV[env_name]
    env_id = int(env_id)
    nb_env = len(li_env)
    if env_id >= nb_env:
        abort(400, custom=f"you asked to run the environment {env_id}  of {env_name}. But there are only "
                          f"{nb_env} such environments")
        return make_response(jsonify({'error': f"you asked to run the environment {env_id}  of {env_name}. "
                                               f"But there are only {nb_env} such environments"}),
                             400)
    env = li_env[env_id]
    # handle the action part
    if not request.json or 'action' not in request.json:
        make_response(jsonify({'error': f"You need to provide an action in order to do a \"step\"."}),
                      400)

    try:
        act = env.action_space()
        act.from_json(request.json["action"])
    except Exception as exc_:
        return make_response(jsonify({'error': f"impossible to convert the provided action to a valid "
                                               f"action on this environment with error:\n"
                                               f"{exc_}"}),
                             400)

    resp = {"id": env_id}
    try:
        obs, reward, done, info = env.step(act)
    except Exception as exc_:
        return make_response(jsonify({'error': f"impossible to make a step on the give environment with "
                                               f"error\n{exc_}"}),
                             400)

    resp["obs"] = obs.to_json()
    resp["reward"] = float(reward)
    resp["done"] = bool(done)
    resp["info"] = to_correct_json(info)
    return make_response(jsonify(resp))


if __name__ == '__main__':
    app.run(debug=True)
