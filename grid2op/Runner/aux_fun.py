# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import time
import warnings
import numpy as np

from grid2op.Environment import Environment
from grid2op.Agent import BaseAgent

from grid2op.Episode import EpisodeData, CompactEpisodeData
from grid2op.Runner.FakePBar import _FakePbar
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Chronics import ChronicsHandler


def _aux_add_data(reward, env, episode,
                  efficient_storing, end__, beg__, act,
                  obs, info, time_step, opp_attack):
    episode.incr_store(
        efficient_storing,
        time_step,
        end__ - beg__,
        float(reward),
        env._env_modification,
        act,
        obs,
        opp_attack,
        info,
    )
    return reward
                
                
def _aux_one_process_parrallel(
    runner,
    episode_this_process,
    process_id,
    path_save=None,
    env_seeds=None,
    agent_seeds=None,
    max_iter=None,
    add_detailed_output=False,
    add_nb_highres_sim=False,
    init_states=None,
    reset_options=None,
):
    """this is out of the runner, otherwise it does not work on windows / macos"""
    parameters = copy.deepcopy(runner.parameters)
    nb_episode_this_process = len(episode_this_process)
    res = [(None, None, None) for _ in range(nb_episode_this_process)]
    for i, ep_id in enumerate(episode_this_process):
        # `ep_id`: grid2op id of the episode i want to play
        # `i`: my id of the episode played (0, 1, ... episode_this_process)
        env, agent = runner._new_env(parameters=parameters)
        try:
            env_seed = None
            if env_seeds is not None:
                env_seed = env_seeds[i]
                
            agt_seed = None
            if agent_seeds is not None:
                agt_seed = agent_seeds[i]
                
            if init_states is not None:
                init_state = init_states[i]
            else:
                init_state = None
            
            if reset_options is not None:
                reset_option = reset_options[i]
            else:
                reset_option = None
            tmp_ = _aux_run_one_episode(
                env,
                agent,
                runner.logger,
                ep_id,
                path_save,
                env_seed=env_seed,
                max_iter=max_iter,
                agent_seed=agt_seed,
                detailed_output=add_detailed_output,
                use_compact_episode_data=runner.use_compact_episode_data,
                init_state=init_state,
                reset_option=reset_option
            )
            (name_chron, cum_reward, nb_time_step, max_ts, episode_data, nb_highres_sim)  = tmp_
            id_chron = env.chronics_handler.get_id()
            res[i] = (id_chron, name_chron, float(cum_reward), nb_time_step, max_ts)
            
            if add_detailed_output:
                res[i] = (*res[i], episode_data)
            if add_nb_highres_sim:
                res[i] = (*res[i], nb_highres_sim)                
        finally:
            env.close()
    return res


def _aux_run_one_episode(
    env: Environment,
    agent: BaseAgent,
    logger,
    indx : int,
    path_save=None,
    pbar=False,
    env_seed=None,
    agent_seed=None,
    max_iter=None,
    detailed_output=False,
    use_compact_episode_data=False,
    init_state=None,
    reset_option=None,
):
    done = False
    time_step = int(0)
    time_act = 0.0
    cum_reward = dt_float(0.0)

    # set the environment to use the proper chronic
    # env.set_id(indx)
    if reset_option is None:
        reset_option = {}
    
    if "time serie id" in reset_option:
        warnings.warn("You provided both `episode_id` and the key `'time serie id'` is present "
                      "in the provided `reset_options`. In this case, grid2op will ignore the "
                      "`time serie id` of the `reset_options` and keep the value in `episode_id`.")
    reset_option["time serie id"] = indx
    
    # handle max_iter
    if max_iter is not None:
        if "max step" in reset_option:
            warnings.warn("You provided both `max_iter` and the key `'max step'` is present "
                          "in the provided `reset_options`. In this case, grid2op will ignore the "
                          "`max step` of the `reset_options` and keep the value in `max_iter`.")
        reset_option["max step"] = max_iter
        
    # handle init state
    if init_state is not None:
        if "init state" in reset_option:
            warnings.warn("You provided both `init_state` and the key `'init state'` is present "
                          "in the provided `reset_options`. In this case, grid2op will ignore the "
                          "`init state` of the `reset_options` and keep the value in `init_state`.")
        reset_option["init state"] = init_state
        
    # reset it
    obs = env.reset(seed=env_seed, options=reset_option)
        
    # reset the number of calls to high resolution simulator
    env._highres_sim_counter._HighResSimCounter__nb_highres_called = 0
    
    # seed and reset the agent
    if agent_seed is not None:
        agent.seed(agent_seed)
    agent.reset(obs)

    # compute the size and everything if it needs to be stored
    nb_timestep_max = env.chronics_handler.max_timestep()
    efficient_storing = nb_timestep_max > 0
    nb_timestep_max = max(nb_timestep_max, 0)
    max_ts = nb_timestep_max
    if use_compact_episode_data:
        episode = CompactEpisodeData(env, obs, exp_dir=path_save)
    else:
        if path_save is None and not detailed_output:
            # i don't store anything on drive, so i don't need to store anything on memory
            nb_timestep_max = 0

        disc_lines_templ = np.full((1, env.backend.n_line), fill_value=False, dtype=dt_bool)

        attack_templ = np.full(
            (1, env._oppSpace.action_space.size()), fill_value=0.0, dtype=dt_float
        )
        
        if efficient_storing:
            times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
            rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
            actions = np.full(
                (nb_timestep_max, env.action_space.n), fill_value=np.NaN, dtype=dt_float
            )
            env_actions = np.full(
                (nb_timestep_max, env._helper_action_env.n),
                fill_value=np.NaN,
                dtype=dt_float,
            )
            observations = np.full(
                (nb_timestep_max + 1, env.observation_space.n),
                fill_value=np.NaN,
                dtype=dt_float,
            )
            disc_lines = np.full(
                (nb_timestep_max, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool
            )
            attack = np.full(
                (nb_timestep_max, env._opponent_action_space.n),
                fill_value=0.0,
                dtype=dt_float,
            )
            legal = np.full(nb_timestep_max, fill_value=True, dtype=dt_bool)
            ambiguous = np.full(nb_timestep_max, fill_value=False, dtype=dt_bool)
        else:
            times = np.full(0, fill_value=np.NaN, dtype=dt_float)
            rewards = np.full(0, fill_value=np.NaN, dtype=dt_float)
            actions = np.full((0, env.action_space.n), fill_value=np.NaN, dtype=dt_float)
            env_actions = np.full(
                (0, env._helper_action_env.n), fill_value=np.NaN, dtype=dt_float
            )
            observations = np.full(
                (0, env.observation_space.n), fill_value=np.NaN, dtype=dt_float
            )
            disc_lines = np.full((0, env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
            attack = np.full(
                (0, env._opponent_action_space.n), fill_value=0.0, dtype=dt_float
            )
            legal = np.full(0, fill_value=True, dtype=dt_bool)
            ambiguous = np.full(0, fill_value=False, dtype=dt_bool)

        need_store_first_act = path_save is not None or detailed_output
        if need_store_first_act:
            # store observation at timestep 0
            if efficient_storing:
                observations[time_step, :] = obs.to_vect()
            else:
                observations = np.concatenate((observations, obs.to_vect().reshape(1, -1)))
                
        episode = EpisodeData(
            actions=actions,
            env_actions=env_actions,
            observations=observations,
            rewards=rewards,
            disc_lines=disc_lines,
            times=times,
            observation_space=env.observation_space,
            action_space=env.action_space,
            helper_action_env=env._helper_action_env,
            path_save=path_save,
            disc_lines_templ=disc_lines_templ,
            attack_templ=attack_templ,
            attack=attack,
            attack_space=env._opponent_action_space,
            logger=logger,
            name=env.chronics_handler.get_name(),
            force_detail=detailed_output,
            other_rewards=[],
            legal=legal,
            ambiguous=ambiguous,
            has_legal_ambiguous=True,
        )
        if need_store_first_act:
            # I need to manually force in the first observation (otherwise it's not computed)
            episode.observations.objects[0] = episode.observations.helper.from_vect(
                observations[time_step, :]
            )
        episode.set_parameters(env)

    beg_ = time.perf_counter()

    reward = float(env.reward_range[0])
    done = False

    next_pbar = [False]
    with _aux_make_progress_bar(pbar, nb_timestep_max, next_pbar) as pbar_:
        while not done:
            beg__ = time.perf_counter()
            act = agent.act(obs, reward, done)
            end__ = time.perf_counter()
            time_act += end__ - beg__
            
            if type(env).CAN_SKIP_TS:
                # the environment can "skip" some time
                # steps I need to call the 'env.steps()' to get all
                # the steps.
                res_env_tmp = env.steps(act)
                for (obs, reward, done, info), opp_attack in zip(*res_env_tmp):
                    time_step += 1
                    if use_compact_episode_data:
                        duration = end__ - beg__
                        cum_reward = episode.update(time_step, env, act,
                                                    obs, reward, done, duration, info)
                    else:
                        cum_reward += _aux_add_data(reward, env, episode,
                                                    efficient_storing,
                                                    end__, beg__, act,
                                                    obs, info, time_step,
                                                    opp_attack)
                    pbar_.update(1)
            else:
                # regular environment
                obs, reward, done, info = env.step(act)
                time_step += 1
                opp_attack = env._oppSpace.last_attack
                if use_compact_episode_data:
                    duration = end__ - beg__
                    cum_reward = episode.update(time_step, env, act,
                                                obs, reward, done, duration, info)
                else:
                    cum_reward += _aux_add_data(reward, env, episode,
                                                efficient_storing,
                                                end__, beg__, act,
                                                obs, info, time_step,
                                                opp_attack)
                pbar_.update(1)
        if not use_compact_episode_data:
            episode.set_game_over(time_step)
        end_ = time.perf_counter()
    if not use_compact_episode_data:
        episode.set_meta(env, time_step, float(cum_reward), env_seed, agent_seed)
    li_text = [
        "Env: {:.2f}s",
        "\t - apply act {:.2f}s",
        "\t - run pf: {:.2f}s",
        "\t - env update + observation: {:.2f}s",
        "Agent: {:.2f}s",
        "Total time: {:.2f}s",
        "Cumulative reward: {:1f}",
    ]
    msg_ = "\n".join(li_text)
    logger.info(
        msg_.format(
            env._time_apply_act + env._time_powerflow + env._time_extract_obs,
            env._time_apply_act,
            env._time_powerflow,
            env._time_extract_obs,
            time_act,
            end_ - beg_,
            cum_reward,
        )
    )
    if not use_compact_episode_data:
        episode.set_episode_times(env, time_act, beg_, end_)

    episode.to_disk()
    name_chron = env.chronics_handler.get_name()
    return (name_chron, cum_reward,
            int(time_step),
            int(max_ts),
            episode,
            env.nb_highres_called)


def _aux_make_progress_bar(pbar, total, next_pbar):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Parameters
    ----------
    pbar: ``bool`` or ``type`` or ``object``
        How to display the progress bar, understood as follow:

        - if pbar is ``None`` nothing is done.
        - if pbar is a boolean, tqdm pbar are used, if tqdm package is available and installed on the system
          [if ``true``]. If it's false it's equivalent to pbar being ``None``
        - if pbar is a ``type`` ( a class), it is used to build a progress bar at the highest level (episode) and
          and the lower levels (step during the episode). If it's a type it muyst accept the argument "total"
          and "desc" when being built, and the closing is ensured by this method.
        - if pbar is an object (an instance of a class) it is used to make a progress bar at this highest level
          (episode) but not at lower levels (step during the episode)
    """
    pbar_ = _FakePbar()
    next_pbar[0] = False

    if isinstance(pbar, bool):
        if pbar:
            try:
                from tqdm import tqdm

                pbar_ = tqdm(total=total, desc="episode")
                next_pbar[0] = True
            except (ImportError, ModuleNotFoundError):
                pass
    elif isinstance(pbar, type):
        pbar_ = pbar(total=total, desc="episode")
        next_pbar[0] = pbar
    elif isinstance(pbar, object):
        pbar_ = pbar
    return pbar_
