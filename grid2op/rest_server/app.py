# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import typing as t
import warnings

from flask import Flask
from flask import make_response, jsonify
from flask import request
from collections.abc import Iterable

from grid2op.rest_server.env_cache import EnvCache
import argparse

try:
    import ujson
    from flask.json import JSONEncoder, JSONDecoder

    # define the encoder
    class CustomJSONEncoder(JSONEncoder):
        def default(self, obj):
            try:
                return ujson.dumps(obj)
            except TypeError:
                return JSONEncoder.default(self, obj)

    # define the decoder
    class CustomJSONDecoder(JSONDecoder):
        def dump(self, obj: t.Any, fp: t.IO[str], app: t.Optional["Flask"] = None, **kwargs: t.Any) -> None:
            try:
                return ujson.dump(obj=obj, fp=fp)
            except TypeError:
                return CustomJSONDecoder.dump(self, obj, fp, app, **kwargs)

        def dumps(self, obj: t.Any, app: t.Optional["Flask"] = None, **kwargs: t.Any) -> str:
            try:
                return ujson.dumps(obj=obj)
            except TypeError:
                return CustomJSONDecoder.dumps(self, obj, app, **kwargs)

        def loads(self, s: str, app: t.Optional["Flask"] = None, **kwargs: t.Any) -> t.Any:
            try:
                return ujson.loads(s)
            except TypeError:
                return CustomJSONDecoder.loads(self, s, app, **kwargs)

        def load(self, fp: t.IO[str], app: t.Optional["Flask"] = None, **kwargs: t.Any) -> t.Any:
            try:
                return ujson.load(fp=fp)
            except TypeError:
                return CustomJSONDecoder.load(self, fp, app, **kwargs)

    UJSON_AS_JSON = True
except ImportError as exc:
    warnings.warn("ujson not available, expect some degraded performance")
    UJSON_AS_JSON = False

ENV_CACHE = EnvCache(UJSON_AS_JSON)

app = Flask(__name__)
if UJSON_AS_JSON:
    app.json_encoder = CustomJSONEncoder
    app.json_decoder = CustomJSONDecoder

# TODO for improved security, not sure it's needed
if False:
    from flask_wtf.csrf import CSRFProtect
    csrf = CSRFProtect()
    csrf.init_app(app)
    # set the env variable this way before starting : `set WTF_CSRF_SECRET_KEY=...`
    SECRET_KEY = os.urandom(32)
    app.config['SECRET_KEY'] = SECRET_KEY


@app.route('/')
def index():
    return "Welcome to grid2op. This small server lets you use grid2op as an web service to use some grid2op " \
           "features for example in different computer languages. See the documentation for more information." \
           "(work in progress)" \
           "(alpha mode at the moment)"


@app.route('/make/<env_name>')
def make_env(env_name):
    """
    This function lets you create an environment with the name "env_name".

    It is equivalent to perform a call to `grid2op.make(env_name)` followed by `obs = env.reset()`

    TODO support parameters and backend and all the other kwargs of make

    Notes
    ------
    This is a simple `get` request.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "obs" : the json representation of the observation you get after creation of this environment

    """
    id_, obs, exc_ = ENV_CACHE.insert_env(env_name)
    if exc_ is not None:
        return make_response(jsonify({'error': f"Impossible to create environment \"{env_name}\" with error:\n"
                                               f"{exc_}"}), 400)
    resp = {"id": id_, "env_name": env_name, "obs": obs}
    return jsonify(resp)


@app.route('/reset/<env_name>/<env_id>')
def reset(env_name, env_id):
    """
    This call is equivalent to do: `env.reset()` when ``env`` is the environment with id "env_id" and name "env_name"

    Notes
    ------
    This is a simple `get` request.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "obs" : the json representation of the observation you get after having reset this environment.

    """
    obs, (error_code, error_msg) = ENV_CACHE.reset(env_name, env_id)
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "obs": obs}
    return make_response(jsonify(resp))


@app.route('/close/<env_name>/<env_id>')
def close(env_name, env_id):
    """
    This call is equivalent to do: `env.close()` when ``env`` is the environment with id "env_id" and name "env_name".

    Note that after being closed, any use of the environment will raise an error.

    Notes
    ------
    This is a simple `get` request.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created

    """
    error_code, error_msg = ENV_CACHE.close(env_name, env_id)
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name}
    return make_response(jsonify(resp))


@app.route('/get_path_env/<env_name>/<env_id>')
def get_path_env(env_name, env_id):
    """
    This call is equivalent to do: `env.get_path_env()` when ``env`` is the environment with id "env_id" and name
    "env_name".

    It returns the (local to the server, not the client!) path where the environment is located.

    Notes
    ------
    This is a simple `get` request.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "path": the path of the environment (**NB** this path is local to the server!)

    """
    path, (error_code, error_msg) = ENV_CACHE.get_path_env(env_name, env_id)
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "path": path}
    return make_response(jsonify(resp))


@app.route('/get_thermal_limit/<env_name>/<env_id>')
def get_thermal_limit(env_name, env_id):
    """
    This call is equivalent to do: `env.get_thermal_limit()` when ``env`` is the environment with id "env_id" and name
    "env_name".

    Notes
    ------
    This is a simple `get` request.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "thermal_limit": ``list`` the thermal limit for each powerline.

    """
    th_lim, (error_code, error_msg) = ENV_CACHE.get_thermal_limit(env_name, env_id)
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "thermal_limit": th_lim}
    return make_response(jsonify(resp))


@app.route('/step/<env_name>/<env_id>', methods=["POST"])
def step(env_name, env_id):
    """
    This call is equivalent to do: `env.step(action)` when ``env`` is the environment with id "env_id" and name
    "env_name".

    Notes
    ------
    This is a `post` request.

    The payload (data) should be a json with key "action" (``dict``) representing a valid grid2op action

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "obs": the json representation of the observation you get after this step
    - "reward": the reward you get after this step (``float``)
    - "done": a flag indicating whether or not the environment has terminated (``bool``). If this flag is
      ``True`` then you need to call `reset` on this same environment (same name, same id)
      if you want to continue to use it.
    - "info": a list of detailed information returned by step (more information in the documentation of
      :func:`grid2op.Environment.BaseEnv.step`)

    """
    # handle the action part
    if not request.json or 'action' not in request.json:
        make_response(jsonify({'error': f"You need to provide an action in order to do a \"step\"."}),
                      400)

    (obs, reward, done, info), (error_code, error_msg) = ENV_CACHE.step(env_name, env_id, request.json["action"])
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "obs": obs, "reward": reward, "done": done, "info": info}
    return make_response(jsonify(resp))


@app.route('/seed/<env_name>/<env_id>', methods=["POST"])
def seed(env_name, env_id):
    """
    This call is equivalent to do: `env.seed(seed)` when ``env`` is the environment with id "env_id" and name
    "env_name".

    Notes
    ------
    This is a `post` request.

    The payload (data) should be a json with key "seed" (``int``) representing the seed (an integer) you want to use.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "seeds": the seeds used to ensure reproducibility of all the environment components
      (more information in the documentation of :func:`grid2op.Environment.BaseEnv.seed`)
    - "info": a generic text to make sure you know that you need to call reset before it has any effect.

    """
    if not request.json or 'seed' not in request.json:
        make_response(jsonify({'error': f"You need to provide an action in order to \"seed\" the environment."}),
                      400)

    seeds, (error_code, error_msg) = ENV_CACHE.seed(env_name, env_id, request.json["seed"])
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "seeds": seeds, "info": "this has no effect until reset() is called"}
    return make_response(jsonify(resp))


@app.route('/set_id/<env_name>/<env_id>', methods=["POST"])
def set_id(env_name, env_id):
    """
    This call is equivalent to do: `env.set_id(id)` when ``env`` is the environment with id "env_id" and name
    "env_name".

    It has no effect unless "reset" is used.

    Notes
    ------
    This is a `post` request.

    The payload (data) should be a json with key "id" (``int``) representing the chronic id you want to go to.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "info": a generic text to make sure you know that you need to call reset before it has any effect.

    """
    if not request.json or 'id' not in request.json:
        make_response(jsonify({'error': f"You need to provide an id in order to \"set_id\" the environment."}),
                      400)
    error_code, error_msg = ENV_CACHE.set_id(env_name, env_id, request.json["id"])
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "info": "this has no effect until reset() is called"}
    return make_response(jsonify(resp))


@app.route('/set_thermal_limit/<env_name>/<env_id>', methods=["POST"])
def set_thermal_limit(env_name, env_id):
    """
    This call is equivalent to do: `env.set_thermal_limit(thermal_limits)` when
    ``env`` is the environment with id "env_id" and name "env_name".

    Notes
    ------
    This is a `post` request.

    The payload (data) should be a json with key "thermal_limits" (``list``) representing the new thermal limit you
    want to use.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created

    """
    if not request.json or 'thermal_limits' not in request.json:
        make_response(jsonify({'error': f"You need to provide thermal limits in order to \"set_thermal_limit\" "
                                        f"the environment."}),
                      400)
    error_code, error_msg = ENV_CACHE.set_thermal_limit(env_name, env_id, request.json["thermal_limits"])
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name}
    return make_response(jsonify(resp))


@app.route('/fast_forward_chronics/<env_name>/<env_id>', methods=["POST"])
def fast_forward_chronics(env_name, env_id):
    """
    This call is equivalent to do: `env.fast_forward_chronics(nb_step)` when
    ``env`` is the environment with id "env_id" and name "env_name".

    Notes
    ------
    This is a `post` request.

    The payload (data) should be a json with key "nb_step" (``int``) representing the number of step you want to
    "fast forward" to.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created

    """
    if not request.json or 'nb_step' not in request.json:
        make_response(jsonify({'error': f"You need to provide a number of step in order to \"fast_forward_chronics\" "
                                        f"the environment."}),
                      400)
    error_code, error_msg = ENV_CACHE.fast_forward_chronics(env_name, env_id, request.json["nb_step"])
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name}
    return make_response(jsonify(resp))


@app.route('/train_val_split/<env_name>/<env_id>', methods=["POST"])
def train_val_split(env_name, env_id):
    """
    This call is equivalent to do: `env.train_val_split(chron_id_val)` when
    ``env`` is the environment with id "env_id" and name "env_name".

    Notes
    ------
    This is a `post` request.

    The payload (data) should be a json with key "chron_id_val" (``list``)
    representing the ids of the chronics that will be put aside in the validation set.

    Returns
    -------
    A json with keys:

    - "id": the id of the environment created
    - "env_name": the name of the environment created
    - "nm_train": name of the environment you can use as training environment, that will contain all the initial
      chronics except the one specified in `chron_id_val`. You may initialize it with `make/nm_train`
    - "nm_val": name of the environment you can use as validation environment, that will contain only the chronics
      ids specified in `chron_id_val`. You may initialize it with `make/nm_val`

    """
    if not request.json or 'chron_id_val' not in request.json:
        make_response(jsonify({'error': f"You need to provide with the id of the chronics that will go to "
                                        f"the validation set "}),
                      400)
    chron_id_val = request.json["chron_id_val"]
    if not isinstance(chron_id_val, Iterable):
        make_response(jsonify({'error': f"\"chron_id_val\"  should be an iterable representing the name of the "
                                        f"scenarios "
                                        f"you want to place in the validation set."}),
                      400)
    (nm_train, nm_val), (error_code, error_msg) = \
        ENV_CACHE.train_val_split(env_name, env_id, chron_id_val)
    if error_code is not None:
        return make_response(jsonify({'error': error_msg, "error_code": error_code}), 400)

    resp = {"id": env_id, "env_name": env_name, "nm_train": nm_train, "nm_val": nm_val}
    return make_response(jsonify(resp))

# TODO
# set_id
# set_thermal_limit
# get_thermal_limit
# fast_forward_chronics
# get_path_env
# close
# train_val_split  # not tested

# TODO
# asynch here!


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a mono thread / mono process grid2op environment server')
    parser.add_argument("--port", type=int, default=3000,
                        help="On which port to start the server (default 3000)")
    parser.add_argument("--debug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Start the flask server in debug mode (default: False).")
    args = parser.parse_args()
    app.run(debug=args.debug, port=args.port)
