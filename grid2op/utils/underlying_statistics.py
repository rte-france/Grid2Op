# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import os
import json
import shutil
import re
import numpy as np

from grid2op.dtypes import dt_float
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Environment import MultiMixEnvironment
from grid2op.Episode import EpisodeData
from grid2op.Reward import BaseReward
from grid2op.Exceptions import Grid2OpException


class EpisodeStatistics(object):
    """
    This class allows to serialize / de serialize some information about the data of a given environment.

    Its use happens in two steps:

    - :func:`EpisodeStatistics.compute` where you run some experiments to generate some data. Be carefull, some data
      (for example obs.a_or, obs.rho etc.) depends on the agent you use! This needs to be performed at least once.
    - :func:`EpisodeStatistics.get` retrieve the stored information and get back a numpy array with each rows
      representing a step.

    Note that it does not check what agent do you use. If you want statistics on more than 1 agent, please use
    the `name_stats` key word attribute when you create the `EpisodeStatistics` object.

    Examples
    ---------
    A basic use of this class is the following:

    .. code-block:: python

        import grid2op
        from grid2op.utils import EpisodeStatistics
        env = grid2op.make()

        stats = EpisodeStatistics(env)

        #################################
        # This need to be done only once
        stats.compute(nb_scenario=100)   # this will take a while to compute in most cases
        ################################

        rhos_, scenario_ids = stats.get("rho")
        load_p_, scenario_ids = stats.get("load_p")

        # do something with them

    If you want some statistics with different agent you might also consider giving some names to the way they are
    saved as follow:

    .. code-block:: python

        import grid2op
        from grid2op.utils import EpisodeStatistics
        from grid2op.Parameters import Parameters
        env = grid2op.make()

        nb_scenario = 8

        # for a example a simple do nothing agent
        stats_dn = EpisodeStatistics(env, name_stats="do_nothing")
        stats_dn.compute(nb_scenario=nb_scenario)   # this will take a while to compute in most cases

        # you can also change the parameters
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        stats_no_overflow = EpisodeStatistics(env, name_stats="no_overflow")
        stats_no_overflow.compute(nb_scenario=nb_scenario, parameters=param)   # this will take a while to compute in most cases

        # or use a different agent
        my_agent = ...  # use any grid2op agent you want here
        stats_custom_agent = EpisodeStatistics(env, name_stats="custom_agent")
        stats_custom_agent.compute(nb_scenario=nb_scenario, agent=my_agent)   # this will take a while to compute in most cases

        # and then you can retrieve the statistics
        rho_dn, ids = stats_dn.get("rho")
        rho_dn_all, ids = stats_no_overflow.get("rho")
        rho_custom_agent, ids = stats_custom_agent.get("rho")

    Notes
    -------
    The observations computed highly depends on the agent and the stochastic part of the environment, such as the
    maintenance or the opponent etc. We highly recommend you to use the env_seeds and agent_seeds keyword arguments
    when using the :func:`EpisodeStatistics.compute` function.

    """

    # TODO NB: name for generator are saved as "prod_p.npz", "prod_v.npz" and "prod_q.npz" and not
    # TODO NB: "gen_p.npz" for backward compatibility.
    SCENARIO_IDS = "scenario_ids.npz"
    SCORES = "scores.npz"
    SCORES_CLEAN = re.sub("\\.npz", "", SCORES)
    KEY_SCORE = "__scores"
    SCORE_FOOTPRINT = ".has_score"
    STATISTICS_FOLDER = "_statistics"
    STATISTICS_FOOTPRINT = ".statistics"
    METADATA = "metadata.json"

    def __init__(self, env, name_stats=None):
        if isinstance(env, MultiMixEnvironment):
            raise RuntimeError("MultiMixEnvironment are not supported at the moment")
        self.env = env
        self.path_env = self.env.get_path_env()
        nm_ = self.get_name_dir(name_stats)
        self.path_save_stats = os.path.join(self.path_env, nm_)
        self.li_attributes = self.env.observation_space.attr_list_vect

    @staticmethod
    def get_name_dir(name_stats):
        """return the name of the folder in which the statistics will be computed"""
        if name_stats is not None:
            nm_ = f"{EpisodeStatistics.STATISTICS_FOLDER}_{name_stats}"
        else:
            nm_ = EpisodeStatistics.STATISTICS_FOLDER
        return nm_

    def get_name_file(self, observation_attribute):
        """get the name of the file that is used to save a given attribute names"""
        if observation_attribute not in self.li_attributes:
            raise RuntimeWarning(
                f'Unknown observation attribute: "{observation_attribute}"'
            )
        # backward compatibility
        if observation_attribute == "gen_p":
            observation_attribute = "prod_p"
        elif observation_attribute == "gen_q":
            observation_attribute = "prod_q"
        elif observation_attribute == "gen_v":
            observation_attribute = "prod_v"
        return f"obs_{observation_attribute}.npz"

    def _delete_if_exists(self, path_tmp, episode_name, saved_stuff):
        full_path = os.path.join(path_tmp, episode_name, saved_stuff)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            os.remove(full_path)

    @staticmethod
    def _save_numpy(path, array):
        np.savez_compressed(path, data=array)

    @staticmethod
    def _load(path):
        return np.load(path)["data"]

    def _clean_observations(self, path_tmp, episode_name):
        full_path = os.path.join(path_tmp, episode_name, EpisodeData.OBSERVATIONS_FILE)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            # this is not a proper path for the observation
            return

        # todo the way to load back the saved data need to be done in episode data instead
        all_obs = np.load(full_path)["data"]
        # handle the end of the episode
        with open(
            os.path.join(path_tmp, episode_name, EpisodeData.META),
            "r",
            encoding="utf-8",
        ) as f:
            metadata_ep = json.load(f)
        nb_ts = int(metadata_ep["nb_timestep_played"]) + 1
        all_obs = all_obs[:nb_ts, :]
        for obs_nm in self.env.observation_space.attr_list_vect:
            beg_, end_, dtype = self.env.observation_space.get_indx_extract(obs_nm)
            all_attr = all_obs[:, beg_:end_].astype(dtype)
            self._save_numpy(
                os.path.join(path_tmp, episode_name, self.get_name_file(obs_nm)),
                all_attr,
            )
        self._delete_if_exists(path_tmp, episode_name, EpisodeData.OBSERVATIONS_FILE)

    def _gather_all(self, li_episodes, dict_metadata, score_names):
        """gather all the data from all the episodes into large array (for easier access later on)"""
        if len(li_episodes) == 0:
            return

        ids_ = np.zeros(shape=(0, 1))
        scores = None
        if score_names:
            scores = {el: None for el in score_names}

        first_attr = True
        for obs_nm in self.li_attributes:
            res = None
            for i, (path_tmp, episode_name) in enumerate(li_episodes):
                # retrieve the content of the attributes
                tmp_arr = self._load(
                    os.path.join(path_tmp, episode_name, self.get_name_file(obs_nm))
                )
                if res is None:
                    res = tmp_arr
                else:
                    res = np.concatenate((res, tmp_arr))

                if first_attr:
                    dict_metadata[f"{i}"] = {
                        "path": path_tmp,
                        "scenario_name": episode_name,
                        "nb_step": int(tmp_arr.shape[0]),
                    }

                # save the ids corresponding to each scenarios (but only once)
                if first_attr:
                    scen_sz = tmp_arr.shape[0]
                    tmp_ids = np.ones(scen_sz, dtype=int).reshape((-1, 1))
                    tmp_ids *= i
                    tmp_ids = tmp_ids.astype(int)
                    ids_ = np.concatenate((ids_, tmp_ids))

                    # handles the scores (same, only once)
                    if score_names:
                        for el in score_names:
                            tmp_scor = self._load(
                                os.path.join(path_tmp, episode_name, el)
                            )
                            if len(score_names) == 0:
                                dict_metadata[f"{i}"]["scores"] = float(
                                    np.sum(tmp_scor)
                                )
                            else:
                                dict_metadata[f"{i}"][f"scores_{el}"] = float(
                                    np.sum(tmp_scor)
                                )

                            if scores[el] is None:
                                scores[el] = tmp_scor
                            else:
                                scores[el] = np.concatenate((scores[el], tmp_scor))

            # save for each attributes its content
            path_total = li_episodes[0][0]
            self._save_numpy(
                os.path.join(path_total, self.get_name_file(obs_nm)), array=res
            )

            # save the id, the metadata and the scores but only once
            if first_attr:
                self._save_numpy(
                    os.path.join(path_total, self.SCENARIO_IDS), array=ids_
                )
                if score_names:
                    for el in scores:
                        self._save_numpy(os.path.join(path_total, el), array=scores[el])
                    del scores
                del ids_
                with open(
                    os.path.join(path_total, EpisodeStatistics.METADATA),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(obj=dict_metadata, fp=f)
            first_attr = False

    @staticmethod
    def list_stats(env):
        """this is a function listing all the stats that have been computed for this environment"""
        res = []
        path_env = env.get_path_env()
        for el in os.listdir(path_env):
            if os.path.exists(
                os.path.join(path_env, el, EpisodeStatistics.STATISTICS_FOOTPRINT)
            ):
                res.append((path_env, el))
        return sorted(res)

    @staticmethod
    def _nm_score_from_attr_name(attribute_name):
        return re.sub(
            f"(_{{0,1}}{EpisodeStatistics.SCORES_CLEAN})|(\\.npz)|(\\.npy)",
            "",
            attribute_name,
        )

    @staticmethod
    def _is_score_attribute(attribute_name):
        nm = None
        # test if it a single stat or not
        has_stat = (
            attribute_name == EpisodeStatistics.SCORES
            or attribute_name == EpisodeStatistics.SCORES_CLEAN
        )
        if has_stat:
            nm = EpisodeStatistics.SCORES
        else:
            # i test if it's a statistics with multiple scores
            if (
                re.match(f".*_{EpisodeStatistics.SCORES_CLEAN}", attribute_name)
                is not None
            ):
                # it's a match: multiple score were computed for this name
                # i need to compute the name with which the files are stored
                nm_stat = EpisodeStatistics._nm_score_from_attr_name(attribute_name)
                has_stat = True
                nm = f"{nm_stat}_{EpisodeStatistics.SCORES}"  # should be the same as in "_retrieve_scores" function
        return has_stat, nm

    def get(self, attribute_name):
        """
        This function supposes that you previously ran the :func:`EpisodeStatistics.compute` to have lots of
        observations.

        It allows the retrieval of the information about the observation that were previously stored on drive.

        Parameters
        ----------
        attribute_name: ``str``
            The name of the attribute of an observation on which you want some information.

        Returns
        -------
        values: ``numpy.ndarray``
            All the values for the "attribute_name" of all the observations that were obtained when running the
            :func:`EpisodeStatistics.compute`. It has the shape (nb step, dim_attribute).

        ids: ``numpy.ndarray``
            The scenario ids to which belong the "values" value. It has the same number of rows than "values" but
            only one column. This unique column contains an integer. If two rows have the same id then they come from
            the same scenario.

        """
        # backward compatibility
        if attribute_name == "prod_p":
            attribute_name = "gen_p"
        elif attribute_name == "prod_q":
            attribute_name = "gen_q"
        elif attribute_name == "prod_v":
            attribute_name = "gen_v"

        if not os.path.exists(self.path_save_stats) or not os.path.isdir(
            self.path_save_stats
        ):
            raise RuntimeError(
                "No statistics were computed for this environment. "
                'Please use "self.compute()" to compute them. '
                "And most importantly have a look at the documentation for precisions about this "
                "feature."
            )
        ids = self._load(
            os.path.join(self.path_save_stats, EpisodeStatistics.SCENARIO_IDS)
        ).astype(int)
        is_score, score_name = EpisodeStatistics._is_score_attribute(attribute_name)
        if is_score:
            if not self._get_has_score():
                # not score have been saved
                raise RuntimeError(
                    'No score have been computed for this statistics. Please re run "stats.compute" '
                    'by setting the "scores_func" argument.'
                )
            # TODO here for multiple score
            path_th = os.path.join(self.path_save_stats, score_name)
            ids_ = np.concatenate((ids[:, 0], (-1,)))
            diff_ = np.diff(ids_)
            ids = ids[diff_ == 0, :]
        else:
            path_th = os.path.join(
                self.path_save_stats, self.get_name_file(attribute_name)
            )
        if not os.path.exists(path_th) or not os.path.isfile(path_th):
            raise RuntimeError(
                f'Impossible to read the statistics for attribute "{attribute_name}"'
            )
        array_ = self._load(path_th)
        return array_, ids

    def clear_episode_data(self):
        """
        Has side effects

        .. warning:: /!\\\\ Be careful /!\\\\

        To save space, it clears the data for each episode.

        This is permanent. If you want this data to be available again, you will need to run an expensive
        :func:`EpisodeStatistics.compute` again.

        Notes
        -----
        It clears all directory into the "statistics" directory

        """
        if not os.path.exists(self.path_save_stats) or not os.path.isdir(
            self.path_save_stats
        ):
            raise RuntimeError(
                "No statistics have been saved for this environment. Please use "
                '"stat.compute" to save some (this might take a while, '
                "see the documentation)"
            )
        for episode_name in sorted(os.listdir(self.path_save_stats)):
            path_tmp = os.path.join(self.path_save_stats, episode_name)
            if os.path.isdir(path_tmp):
                shutil.rmtree(path_tmp)

    def clear_all(self):
        """
        Has side effects

        .. warning:: /!\\\\ Be careful /!\\\\

        Clear the whole statistics directory.

        This is permanent. If you want this data to be available again, you will need to run an expensive
        :func:`EpisodeStatistics.compute` again.

        Once done, this cannot be undone.
        """
        if os.path.exists(self.path_save_stats) and os.path.isdir(self.path_save_stats):
            shutil.rmtree(self.path_save_stats)

    @staticmethod
    def clean_all_stats(env):
        """
        Has possibly huge side effects

        .. warning:: /!\\\\ Be extremely careful /!\\\\

        This function cleans all the statistics that have been computed for this environment.

        This cannot be undone is permanent and is equivalent to calling :func:`EpisodeStatistics.clear_all` on all
        statistics ever computed on this episode.

        """
        li_stats = EpisodeStatistics.list_stats(env)
        for path, el in li_stats:
            shutil.rmtree(os.path.join(path, el))

    def _tell_is_stats(self):
        """put the footprint to inform grid2op this is a stat directory"""
        path_tmp = os.path.join(
            self.path_save_stats, EpisodeStatistics.STATISTICS_FOOTPRINT
        )
        with open(path_tmp, "w", encoding="utf-8") as f:
            f.write(
                "This files is internal to grid2op. Expect some inconsistent behaviour if you attempt to modify "
                "it, remove it, alter it in any ways, copy it in another directory etc.\n"
            )

    def _tell_has_score(self):
        """put the footprint to inform grid2op this is a stat directory"""
        path_tmp = os.path.join(self.path_save_stats, EpisodeStatistics.SCORE_FOOTPRINT)
        with open(path_tmp, "w", encoding="utf-8") as f:
            f.write(
                "This files is internal to grid2op. Expect some inconsistent behaviour if you attempt to modify "
                "it, remove it, alter it in any ways, copy it in another directory etc.\n"
            )

    def _get_has_score(self):
        """say if a score has been computed or not"""
        res = os.path.exists(
            os.path.join(self.path_save_stats, EpisodeStatistics.SCORE_FOOTPRINT)
        )
        if res:
            res = os.path.isfile(
                os.path.join(self.path_save_stats, EpisodeStatistics.SCORE_FOOTPRINT)
            )
        return res

    def _fill_metadata(self, agent, parameters, max_step, agent_seeds, env_seeds):
        dict_metadata = {}
        dict_metadata["agent_type"] = f"{type(agent)}"
        if agent_seeds is None:
            dict_metadata["agent_seeds"] = None
        else:
            dict_metadata["agent_seeds"] = [int(el) for el in agent_seeds]
        if env_seeds is None:
            dict_metadata["env_seeds"] = None
        else:
            dict_metadata["env_seeds"] = [int(el) for el in env_seeds]
        dict_metadata["max_step"] = int(max_step)
        dict_metadata["parameters"] = parameters.to_dict()
        return dict_metadata

    def _retrieve_scores(self, path_tmp, episode_name):
        my_path = os.path.join(path_tmp, episode_name, EpisodeData.OTHER_REWARDS)
        with open(my_path, "r", encoding="utf-8") as f:
            dict_rewards = json.load(f)

        if not len(dict_rewards):
            # nothing to do if the dictionary is empty
            return

        # check if the score is unique or if there are multiple scores
        tmp = dict_rewards[0]

        if self.KEY_SCORE in tmp:
            # only one score was used
            arr_ = np.array([dt_float(el[self.KEY_SCORE]) for el in dict_rewards])
            self._save_numpy(os.path.join(path_tmp, episode_name, self.SCORES), arr_)
        else:
            for possible_key in tmp:
                if re.match(f"{self.KEY_SCORE }_.*", possible_key) is None:
                    # this key does not represent a score
                    continue
                nm_score = re.sub(f"{self.KEY_SCORE }_", "", possible_key)
                arr_ = np.array([dt_float(el[possible_key]) for el in dict_rewards])
                self._save_numpy(
                    os.path.join(path_tmp, episode_name, f"{nm_score}_{self.SCORES}"),
                    arr_,
                )

    @staticmethod
    def _check_if_base_reward(stuff):
        if isinstance(stuff, type):
            return issubclass(stuff, BaseReward)
        else:
            return isinstance(stuff, BaseReward)

    @staticmethod
    def run_env(
        env,
        path_save,
        parameters,
        scores_func,
        agent,
        nb_scenario,
        max_step,
        env_seeds,
        agent_seeds,
        pbar,
        nb_process,
    ):

        if scores_func is not None:
            if not (
                EpisodeStatistics._check_if_base_reward(scores_func)
                or isinstance(scores_func, dict)
            ):
                raise Grid2OpException(
                    "score_func should be either a dictionary or an instance of BaseReward"
                )

        dict_kwg = env.get_params_for_runner()
        dict_kwg["parameters_path"] = parameters.to_dict()
        if "other_rewards" not in dict_kwg:
            dict_kwg["other_rewards"] = {}
        if scores_func is not None:
            if EpisodeStatistics._check_if_base_reward(scores_func):
                dict_kwg["other_rewards"][EpisodeStatistics.KEY_SCORE] = scores_func
            elif isinstance(scores_func, dict):
                for nm, score_fun in scores_func.items():
                    dict_kwg["other_rewards"][
                        f"{EpisodeStatistics.KEY_SCORE}_{nm}"
                    ] = score_fun
            else:
                raise RuntimeError(
                    '"scores_func" should inherit from "grid2op.Reward.BaseReward" or '
                    "be a dictionary"
                )
        runner = Runner(**dict_kwg, agentClass=None, agentInstance=agent)
        runner.run(
            path_save=path_save,
            nb_episode=nb_scenario,
            max_iter=max_step,
            env_seeds=env_seeds,
            agent_seeds=agent_seeds,
            pbar=pbar,
            nb_process=nb_process,
        )

    def get_metadata(self):
        """return the metadata as a dictionary"""
        with open(
            os.path.join(self.path_save_stats, self.METADATA), "r", encoding="utf-8"
        ) as f:
            res = json.load(f)
        return res

    def compute(
        self,
        agent=None,
        parameters=None,
        nb_scenario=1,
        scores_func=None,
        max_step=-1,
        env_seeds=None,
        agent_seeds=None,
        nb_process=1,
        pbar=False,
    ):
        """
        This function will save (to be later used with :func:`EpisodeStatistics.get_statistics`) all the observation
        at all time steps, for a given number of scenario (see attributes nb_scenario).

        This is useful when you want to store at a given place some information to use later on on your agent.

        Notes
        -----
        Depending on its parameters (mainly the environment, the agent and the number of scenarios computed)
        this function might take a really long time to compute.

        However you only need to compute it once (unless you delete its results with
        :func:`EpisodeStatistics.clear_all` or :func:`EpisodeStatistics.clear_episode_data`

        Results might also take a lot of space on the hard drive (possibly few GB as all information of all
        observations encountered are stored)

        Parameters
        ----------

        agent: :class:`grid2op.Agent.BaseAgent`
            The agent you want to use to generate the statistics. Note that the statistics are highly dependant on
            the agent. For now only one set of statistics are computed. If you want to run a different agent previous
            results will be erased.

        parameters: :class:`grid2op.Parameters.Parameters`
            The parameters you want to use when computing this statistics

        nb_scenario: ``int``
            Number of scenarios that will be evaluated

        scores_func: :class:`grid2op.Reward.BaseReward`
            A reward used to compute the score of an Agent (it can now be a dictionary of BaseReward)

        nb_scenario: ``int``
            On how many scenarios you want the statistics to be computed

        max_step: ``int``
            Maximum number of steps you want to compute (see :func:`grid2op.Runner.Runner.run`)

        env_seeds: ``list``
            List of seeds used for the environment (for reproducible results) (see :func:`grid2op.Runner.Runner.run`)

        agent_seeds: ``list``
            List of seeds used for the agent (for reproducible results) (see :func:`grid2op.Runner.Runner.run`).

        nb_process: ``int``
            Number of process to use (see :func:`grid2op.Runner.Runner.run`)

        pbar: ``bool``
            Whether a progress bar is displayed (see :func:`grid2op.Runner.Runner.run`)

        """
        if agent is None:
            agent = DoNothingAgent(self.env.action_space)
        if parameters is None:
            parameters = copy.deepcopy(self.env.parameters)
        if not isinstance(agent, BaseAgent):
            raise RuntimeError(
                '"agent" should be either "None" to use DoNothingAgent or an agent that inherits '
                "grid2op.Agent.BaseAgent"
            )
        if not isinstance(parameters, Parameters):
            raise RuntimeError(
                '"parameters" should be either "None" to use the default parameters passed in the '
                "environment or inherits grid2op.Parameters.Parameters"
            )

        score_names = None
        dict_metadata = self._fill_metadata(
            agent, parameters, max_step, agent_seeds, env_seeds
        )

        if scores_func is not None:
            if EpisodeStatistics._check_if_base_reward(scores_func):
                dict_metadata["score_class"] = f"{scores_func}"
                score_names = [self.SCORES]
            elif isinstance(scores_func, dict):
                score_names = []
                for nm, score_fun in scores_func.items():
                    if not EpisodeStatistics._check_if_base_reward(score_fun):
                        raise Grid2OpException(
                            'if using "score_fun" as a dictionary, each value need to be a '
                            "BaseReward"
                        )
                    dict_metadata[f"score_class_{nm}"] = f"{score_fun}"
                    score_names.append(f"{nm}_{self.SCORES}")
            else:
                raise Grid2OpException(
                    "score_func should be either a dictionary or an instance of BaseReward"
                )

        self.run_env(
            env=self.env,
            path_save=self.path_save_stats,
            parameters=parameters,
            scores_func=scores_func,
            agent=agent,
            max_step=max_step,
            env_seeds=env_seeds,
            agent_seeds=agent_seeds,
            pbar=pbar,
            nb_process=nb_process,
            nb_scenario=nb_scenario,
        )

        # inform grid2op this is a statistics directory
        self._tell_is_stats()
        if scores_func is not None:
            self._tell_has_score()

        # now clean a bit the output directory
        os.remove(os.path.join(self.path_save_stats, EpisodeData.ACTION_SPACE))
        os.remove(os.path.join(self.path_save_stats, EpisodeData.ATTACK_SPACE))
        os.remove(os.path.join(self.path_save_stats, EpisodeData.ENV_MODIF_SPACE))
        os.remove(os.path.join(self.path_save_stats, EpisodeData.OBS_SPACE))

        li_episodes = EpisodeData.list_episode(self.path_save_stats)
        for path_tmp, episode_name in li_episodes:
            # remove the useless information (saved but not used)
            self._delete_if_exists(path_tmp, episode_name, EpisodeData.ACTIONS_FILE)
            self._delete_if_exists(path_tmp, episode_name, EpisodeData.AG_EXEC_TIMES)
            self._delete_if_exists(path_tmp, episode_name, EpisodeData.LINES_FAILURES)
            self._delete_if_exists(path_tmp, episode_name, EpisodeData.ENV_ACTIONS_FILE)
            self._delete_if_exists(path_tmp, episode_name, EpisodeData.ATTACK)
            if scores_func is not None:
                self._retrieve_scores(path_tmp, episode_name)
            else:
                self._delete_if_exists(
                    path_tmp, episode_name, EpisodeData.OTHER_REWARDS
                )
            self._delete_if_exists(path_tmp, episode_name, EpisodeData.REWARDS)

            # reformat the observation into a proper "human readable" format
            self._clean_observations(path_tmp, episode_name)

        # and now gather the information for at the top level
        self._gather_all(li_episodes, dict_metadata, score_names=score_names)


if __name__ == "__main__":
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Agent import RandomAgent
    from grid2op.Reward import L2RPNSandBoxScore, AlarmReward

    # env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
    nb_scenario = 2

    # # for a example a simple do nothing agent
    # stats_dn = EpisodeStatistics(env, name_stats="do_nothing")
    # stats_dn.compute(nb_scenario=nb_scenario,
    #                  pbar=True,
    #                  scores_func=L2RPNSandBoxScore)  # this will take a while to compute in most cases
    # stats_dn.clear_episode_data()
    #
    # # you can also change the parameters
    # param = Parameters()
    # param.NO_OVERFLOW_DISCONNECTION = True
    # stats_no_overflow = EpisodeStatistics(env, name_stats="no_overflow")
    # stats_no_overflow.compute(nb_scenario=nb_scenario,
    #                           parameters=param,
    #                           pbar=True,
    #                           scores_func=L2RPNSandBoxScore)  # this will take a while to compute in most cases
    # stats_no_overflow.clear_episode_data()
    #
    # # or use a different agent
    # my_agent = RandomAgent(env.action_space)  # use any grid2op agent you want here
    # stats_custom_agent = EpisodeStatistics(env, name_stats="custom_agent")
    # stats_custom_agent.compute(nb_scenario=nb_scenario,
    #                            agent=my_agent,
    #                            pbar=True,
    #                            scores_func=L2RPNSandBoxScore)  # this will take a while to compute in most cases
    # stats_custom_agent.clear_episode_data()
    #
    # # and then you can retrieve the statistics
    # rho_dn, ids = stats_dn.get("rho")
    # rho_dn_all, ids = stats_no_overflow.get("rho")
    # rho_custom_agent, ids = stats_custom_agent.get("rho")

    # with multiple "scores"
    env = grid2op.make(
        "/home/benjamin/Documents/grid2op_dev/grid2op/data_test/l2rpn_neurips_2020_track1_with_alarm",
        backend=LightSimBackend(),
    )
    stats_dn = EpisodeStatistics(env, name_stats="do_nothing")
    stats_dn.compute(
        nb_scenario=nb_scenario,
        pbar=True,
        scores_func={
            "grid_operational_cost": L2RPNSandBoxScore,
            "operator_attention": AlarmReward,
        },
    )
    # rho_dn, ids = stats_dn.get("rho")
    score_op_cost, ids = stats_dn.get("grid_operational_cost_scores")
    score_att_cost, ids = stats_dn.get("operator_attention_scores")
    import pdb

    pdb.set_trace()
    assert score_att_cost.shape[0] == ids.shape[0]
