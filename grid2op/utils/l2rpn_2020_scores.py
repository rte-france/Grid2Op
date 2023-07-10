# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np
import json
import copy
import tempfile

from grid2op.dtypes import dt_float
from grid2op.Reward import L2RPNSandBoxScore
from grid2op.Agent import RecoPowerlineAgent
from grid2op.utils.underlying_statistics import EpisodeStatistics
from grid2op.Episode import EpisodeData

import re


class ScoreL2RPN2020(object):
    """
    This class allows to compute the same score as the one computed for the L2RPN 2020 competitions.

    It uses some "EpisodeStatistics" of the environment to compute these scores. These statistics, if not available
    are computed at the initialization.

    When using it a second time these information are reused.

    Examples
    ---------
    This class can be used as follow:

    .. code-block:: python

        import grid2op
        from grid2op.utils import ScoreL2RPN2020
        from grid2op.Agent import DoNothingAgent

        env = grid2op.make("l2rpn_case14_sandbox")
        nb_scenario = 2
        my_score = ScoreL2RPN2020(env,
                                  nb_scenario=nb_scenario,
                                  env_seeds=[0 for _ in range(nb_scenario)],
                                  agent_seeds=[0 for _ in range(nb_scenario)]
                                  )

        my_agent = DoNothingAgent(env.action_space)
        print(my_score.get(my_agent))


    Notes
    -------
    To prevent overfitting, we strongly recommend you to use the :func:`grid2op.Environment.Environment.train_val_split`
    and use this function on the built validation set only.

    Also note than computing the statistics, and evaluating an agent on a whole dataset of multiple GB can take a
    really long time and a lot of memory. This fact, again, plea in favor of using this function only on
    a validation set.

    We also strongly recommend to set the seeds of your agent (agent_seeds)
    and of the environment (env_seeds) if you want to use this feature. Reproducibility is really important if you
    want to make progress.

    .. warning::

        The triggering (or not) of the recomputation of the statistics is not perfect for now.
        We recommend you to use always
        the same seeds (`env_seeds` and `agent_seeds` key word argument of this functions)
        and the same parameters (`env.parameters`) when using a given environments.

        You might need to clean it manually if you change
        one of theses things by calling :func:`ScoreL2RPN2020.clear_all()` function .

    """

    NAME_DN = "l2rpn_dn"
    NAME_RP_NO_OVERFLOW = "l2rpn_no_overflow_reco"

    def __init__(
        self,
        env,
        env_seeds=None,
        agent_seeds=None,
        nb_scenario=16,
        min_losses_ratio=0.8,
        verbose=0,
        max_step=-1,
        nb_process_stats=1,
        scores_func=L2RPNSandBoxScore,
        score_names=None,
        add_nb_highres_sim=False,
    ):
        self.env = env
        self.nb_scenario = nb_scenario
        self.env_seeds = env_seeds
        self.agent_seeds = agent_seeds
        self.min_losses_ratio = min_losses_ratio
        self.verbose = verbose
        self.max_step = max_step

        computed_scenarios = [el[1] for el in EpisodeStatistics.list_stats(self.env)]

        self.scores_func = scores_func

        # check if i need to compute stat for do nothing
        self.stat_dn = EpisodeStatistics(self.env, self.NAME_DN)
        self._recomputed_dn = self._init_stat(
            self.stat_dn,
            self.NAME_DN,
            computed_scenarios,
            nb_process_stats=nb_process_stats,
            score_names=score_names,
        )

        # check if i need to compute that for do nothing without overflow disconnection
        param_no_overflow = copy.deepcopy(env.parameters)
        param_no_overflow.NO_OVERFLOW_DISCONNECTION = True
        # check if i need to compute that for reco powerline without overflow disconnection
        self.stat_no_overflow_rp = EpisodeStatistics(self.env, self.NAME_RP_NO_OVERFLOW)
        agent_reco = RecoPowerlineAgent(self.env.action_space)
        self._recomputed_no_ov_rp = self._init_stat(
            self.stat_no_overflow_rp,
            self.NAME_RP_NO_OVERFLOW,
            computed_scenarios,
            parameters=param_no_overflow,
            nb_process_stats=nb_process_stats,
            agent=agent_reco,
            score_names=score_names,
        )
        self.add_nb_highres_sim = add_nb_highres_sim
        self.__cleared = False 

    def _init_stat(
        self,
        stat,
        stat_name,
        computed_scenarios,
        parameters=None,
        nb_process_stats=1,
        agent=None,
        score_names=None,
    ):
        """will check if the statistics need to be computed"""
        need_recompute = True
        if score_names is None:
            score_names = [EpisodeStatistics.SCORES]
        if EpisodeStatistics.get_name_dir(stat_name) in computed_scenarios:
            # the things have been computed i check if the number of scenarios is big enough
            scores, ids_ = stat.get(score_names[0])
            metadata = stat.get_metadata()
            max_id = np.max(ids_)

            # i need to recompute if if i did not compute enough scenarios
            need_recompute = max_id < self.nb_scenario - 1

            # if max
            computed_step = int(metadata["max_step"])
            if computed_step > 0:
                # if i have computed the data with
                if self.max_step == -1:
                    # i need to compute now all the dataset, so yes i have to recompute it
                    need_recompute = True

                # i need to recompute only if i ask more steps than what was computed
                need_recompute = need_recompute or self.max_step > metadata["max_step"]

            # TODO check for the seeds here too
            # TODO and check for the class of the scores too
            # TODO check for the parameters too...

        if need_recompute:
            # i need to compute it
            if self.verbose >= 1:
                print(
                    "I need to recompute the statistics for this environment. This will take a while"
                )  # TODO logger
            stat.compute(
                nb_scenario=self.nb_scenario,
                pbar=self.verbose >= 2,
                env_seeds=self.env_seeds,
                agent_seeds=self.agent_seeds,
                scores_func=self.scores_func,
                max_step=self.max_step,
                parameters=parameters,
                nb_process=nb_process_stats,
                agent=agent,
            )
            stat.clear_episode_data()
        return need_recompute

    def _compute_episode_score(
        self,
        ep_id,  # the ID here, which is an integer and is not the ID from chronics balblabla
        meta,
        other_rewards,
        dn_metadata,
        no_ov_metadata,
        score_file_to_use=None,
    ):
        """
        Performs the rescaling of the score given the information stored in the "statistics" of this
        environment.
        """
        # load_p, ids = self.stat_no_overflow.get("load_p")
        # prod_p, _ = self.stat_no_overflow.get("prod_p")
        load_p_rp, ids_rp = self.stat_no_overflow_rp.get("load_p")
        prod_p_rp, _ = self.stat_no_overflow_rp.get("load_p")

        if score_file_to_use is None:
            score_file_to_use = EpisodeStatistics.SCORES
            key_score_file = EpisodeStatistics.KEY_SCORE
        else:
            # should match underlying_statistics.run_env `dict_kwg["other_rewards"][XXX] = ...`
            # XXX is right now f"{EpisodeStatistics.KEY_SCORE}_{nm}" [this should match the XXX]
            real_nm = EpisodeStatistics._nm_score_from_attr_name(score_file_to_use)
            key_score_file = f"{EpisodeStatistics.KEY_SCORE}_{real_nm}"

        scores_dn, ids_dn_sc = self.stat_dn.get(score_file_to_use)
        # scores_no_ov, ids_noov_sc = self.stat_no_overflow.get(score_file_to_use)
        scores_no_ov_rp, ids_noov_sc_rp = self.stat_no_overflow_rp.get(
            score_file_to_use
        )

        # reshape to have 1 dim array
        ids = ids_rp.reshape(-1)
        ids_dn_sc = ids_dn_sc.reshape(-1)
        ids_noov_sc_rp = ids_noov_sc_rp.reshape(-1)

        # there is a hugly "1" at the end of each scores due to the "game over" (or end of game), so i remove it
        scores_dn = scores_dn[ids_dn_sc == ep_id][:-1]
        scores_no_ov_rp = scores_no_ov_rp[ids_noov_sc_rp == ep_id][:-1]

        dn_this = dn_metadata[f"{ep_id}"]
        no_ov_this = no_ov_metadata[f"{ep_id}"]

        n_played = int(meta["nb_timestep_played"])
        dn_step_played = dn_this["nb_step"] - 1
        total_ts = no_ov_this["nb_step"] - 1

        ep_marginal_cost = np.max(self.env.gen_cost_per_MW).astype(dt_float)
        min_losses_ratio = self.min_losses_ratio

        # remember that first observation do not count (it's generated by the environment)
        ep_loads = np.sum(load_p_rp[ids == ep_id, :], axis=1)[1:]
        ep_losses = np.sum(prod_p_rp[ids == ep_id, :], axis=1)[1:] - ep_loads

        if self.max_step > 0:
            scores_dn = scores_dn[: self.max_step]
            # scores_no_ov = scores_no_ov[:self.max_step]
            scores_no_ov_rp = scores_no_ov_rp[: self.max_step]
            ep_loads = ep_loads[: self.max_step]
            ep_losses = ep_losses[: self.max_step]

        # do nothing operationnal cost
        ep_do_nothing_operat_cost = np.sum(scores_dn)
        ep_do_nothing_operat_cost += (
            np.sum(ep_loads[dn_step_played:]) * ep_marginal_cost
        )

        # no overflow disconnection cost
        ep_do_nothing_nodisc_cost = np.sum(scores_no_ov_rp)

        # this agent cumulated operationnal cost
        # same as above: i remove the last element which correspond to the last state, so irrelevant
        ep_cost = np.array([el[key_score_file] for el in other_rewards]).astype(
            dt_float
        )
        if dn_metadata["max_step"] == self.max_step:
            ep_cost = ep_cost[:-1]
        ep_cost = np.sum(ep_cost)
        ep_cost += np.sum(ep_loads[n_played:]) * ep_marginal_cost

        # Compute ranges
        worst_operat_cost = (
            np.sum(ep_loads) * ep_marginal_cost
        )  # operational cost corresponding to the min score
        zero_operat_score = ep_do_nothing_operat_cost
        nodisc_oeprat_cost = ep_do_nothing_nodisc_cost
        best_score = (
            np.sum(ep_losses) * min_losses_ratio
        )  # operational cost corresponding to the max score

        # Linear interp episode reward to codalab score
        if zero_operat_score != nodisc_oeprat_cost:
            # DoNothing agent doesnt complete the scenario
            reward_range = [
                best_score,
                nodisc_oeprat_cost,
                zero_operat_score,
                worst_operat_cost,
            ]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_score, zero_operat_score, worst_operat_cost]
            score_range = [100.0, 0.0, -100.0]
        ep_score = np.interp(ep_cost, reward_range, score_range)
        return ep_score, n_played, total_ts

    def clear_all(self):
        """
        Has side effects

        .. warning:: /!\\\\ Be careful /!\\\\

        Clear the whole statistics directory for the 3 different computed statistics used for the score. It will
        remove the previously computed statistics.

        Once done, this cannot be undone.
        """
        # self.stat_no_overflow.clear_all()
        self.stat_no_overflow_rp.clear_all()
        self.stat_dn.clear_all()
        self.__cleared = True

    def get(self, agent, path_save=None, nb_process=1):
        """
        Get the score of the agent depending on what has been computed.

        TODO The plots will be done later.

        Parameters
        ----------
        agent: :class:`grid2op.Agent.BaseAgent`
            The agent you want to score

        path_save: ``str``
            the path were you want to store the logs of your agent.

        nb_process: ``int``
            Number of process to use for the evaluation

        Returns
        -------
        all_scores: ``list``
            List of the score of your agent per scenarios

        ts_survived: ``list``
            List of the number of step your agent successfully managed for each scenario

        total_ts: ``list``
            Total number of step for each scenario
        """
        if  self.__cleared:
            raise RuntimeError(EpisodeStatistics.ERROR_MSG_CLEANED)
        
        if path_save is not None:
            need_delete = False  # TODO this is soooo dirty
            path_save = os.path.abspath(path_save)
        else:
            need_delete = True
            dir_tmp = tempfile.TemporaryDirectory()
            path_save = dir_tmp.name
            if self.verbose >= 1:
                print("Using a temp directory to store the intermediate data.")  # TODO logger

        if self.verbose >= 1:
            print("Starts the evaluation of the agent")  # TODO logger
        nb_highres_sim = EpisodeStatistics.run_env(
            self.env,
            env_seeds=self.env_seeds,
            agent_seeds=self.agent_seeds,
            path_save=path_save,
            parameters=self.env.parameters,
            scores_func=self.scores_func,
            agent=agent,
            max_step=self.max_step,
            nb_scenario=self.nb_scenario,
            pbar=self.verbose >= 2,
            nb_process=nb_process,
            add_nb_highres_sim=self.add_nb_highres_sim,
        )
        # NB nb_highres_sim is None if self.add_nb_highres_sim is False !
        if self.verbose >= 1:
            print("Start the evaluation of the scores")  # TODO logger

        meta_data_dn = self.stat_dn.get_metadata()
        no_ov_metadata = self.stat_no_overflow_rp.get_metadata()

        all_scores = []
        ts_survived = []
        total_ts = []
        for ep_id in range(self.nb_scenario):
            this_ep_nm = meta_data_dn[f"{ep_id}"]["scenario_name"]
            with open(
                os.path.join(path_save, this_ep_nm, EpisodeData.META),
                "r",
                encoding="utf-8",
            ) as f:
                this_epi_meta = json.load(f)
            with open(
                os.path.join(path_save, this_ep_nm, EpisodeData.OTHER_REWARDS),
                "r",
                encoding="utf-8",
            ) as f:
                this_epi_scores = json.load(f)
            score_this_ep, nb_ts_survived, total_ts_tmp = self._compute_episode_score(
                ep_id,
                meta=this_epi_meta,
                other_rewards=this_epi_scores,
                dn_metadata=meta_data_dn,
                no_ov_metadata=no_ov_metadata,
            )
            all_scores.append(score_this_ep)
            ts_survived.append(nb_ts_survived)
            total_ts.append(total_ts_tmp)

        if need_delete:
            dir_tmp.cleanup()
        res = all_scores, ts_survived, total_ts
        if self.add_nb_highres_sim:
            res = all_scores, ts_survived, total_ts, nb_highres_sim
        return res


if __name__ == "__main__":
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Agent import RandomAgent, DoNothingAgent

    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
    nb_scenario = 16
    my_score = ScoreL2RPN2020(
        env,
        nb_scenario=nb_scenario,
        env_seeds=[0 for _ in range(nb_scenario)],
        agent_seeds=[0 for _ in range(nb_scenario)],
    )

    my_agent = RandomAgent(env.action_space)
    my_agent = DoNothingAgent(env.action_space)
    print(my_score.get(my_agent))
