# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.utils.l2rpn_2020_scores import ScoreL2RPN2020
from grid2op.Reward import L2RPNSandBoxScore, _NewRenewableSourcesUsageScore, _AlertTrustScore
from grid2op.utils.underlying_statistics import EpisodeStatistics
from grid2op.Exceptions import Grid2OpException


class ScoreL2RPN2023(ScoreL2RPN2020):
    """
    This class allows to compute the same score as the one computed for the L2RPN 2023 competitions.

    It uses some "EpisodeStatistics" of the environment to compute these scores. These statistics, if not available
    are computed at the initialization.

    When using it a second time these information are reused.


    This scores is the combination of the `ScoreL2RPN2020` score and some extra scores based on the assistant feature
    (alert) and the use of new renewable energy sources.

    Examples
    ---------
    This class can be used as follow:

    .. code-block:: python

        import grid2op
        from grid2op.utils import ScoreL2RPN2023
        from grid2op.Agent import DoNothingAgent

        env = grid2op.make("l2rpn_case14_sandbox")
        nb_scenario = 2
        my_score = ScoreL2RPN2023(env,
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
            scores_func={
                "grid_operational_cost": L2RPNSandBoxScore,
                "assistant_confidence": _AlertTrustScore,
                "new_renewable_sources_usage": _NewRenewableSourcesUsageScore,
            },
            score_names=["grid_operational_cost_scores",
                         "assistant_confidence_scores",
                         "new_renewable_sources_usage_scores"],
            add_nb_highres_sim=False,
            scale_assistant_score=100.0,
            scale_nres_score=100.,
            weight_op_score=0.6,
            weight_assistant_score=0.25,
            weight_nres_score=0.15,
            min_nres_score=-100,
            min_assistant_score=-300
    ):
        ScoreL2RPN2020.__init__(
            self,
            env=env,
            env_seeds=env_seeds,
            agent_seeds=agent_seeds,
            nb_scenario=nb_scenario,
            min_losses_ratio=min_losses_ratio,
            verbose=verbose,
            max_step=max_step,
            nb_process_stats=nb_process_stats,
            scores_func=scores_func,
            score_names=score_names,
            add_nb_highres_sim=add_nb_highres_sim,
        )
        weights=np.array([weight_op_score, weight_assistant_score, weight_nres_score])
        total_weights = weights.sum()
        if abs(total_weights - 1.0) >= 1e-8:
            raise Grid2OpException(
                'The weights of each component of the score shall sum to 1'
            )
        if np.any(weights < 0.):
            raise Grid2OpException(
                'All weights should be positive'
            )

        self.scale_assistant_score = scale_assistant_score
        self.scale_nres_score = scale_nres_score
        self.weight_op_score = weight_op_score
        self.weight_assistant_score = weight_assistant_score
        self.weight_nres_score = weight_nres_score
        self.min_nres_score = min_nres_score
        self.min_assistant_score = min_assistant_score

    def _compute_episode_score(
            self,
            ep_id,  # the ID here, which is an integer and is not the ID from chronics balblabla
            meta,
            other_rewards,
            dn_metadata,
            no_ov_metadata,
            score_file_to_use="grid_operational_cost_scores",
    ):
        """
        Performs the rescaling of the score given the information stored in the "statistics" of this
        environment.

        This computes the score for a single episode. The loop to compute the score for all the
        episodes is the same as for l2rpn_2020_scores and is then reused.
        """

        # compute the operational score
        op_score, n_played, total_ts = super()._compute_episode_score(
            ep_id,
            meta,
            other_rewards,
            dn_metadata,
            no_ov_metadata,
            # score_file_to_use should match the
            # L2RPNSandBoxScore key in
            # self.scores_func
            score_file_to_use=score_file_to_use,
        )
        # should match underlying_statistics.run_env `dict_kwg["other_rewards"][XXX] = ...`
        # XXX is right now f"{EpisodeStatistics.KEY_SCORE}_{nm}" [this should match the XXX]

        # retrieve nres_score
        new_renewable_sources_usage_score_nm = "new_renewable_sources_usage_scores"
        real_nm = EpisodeStatistics._nm_score_from_attr_name(new_renewable_sources_usage_score_nm)
        key_score_file = f"{EpisodeStatistics.KEY_SCORE}_{real_nm}"
        nres_score = float(other_rewards[-1][key_score_file])
        nres_score = max(nres_score, self.min_nres_score / self.scale_nres_score)
        nres_score = self.scale_nres_score * nres_score

        # assistant_score
        assistant_confidence_score_nm = "assistant_confidence_scores"
        real_nm = EpisodeStatistics._nm_score_from_attr_name(assistant_confidence_score_nm)
        key_score_file = f"{EpisodeStatistics.KEY_SCORE}_{real_nm}"
        assistant_confidence_score = float(other_rewards[-1][key_score_file])
        assistant_confidence_score = max(assistant_confidence_score, self.min_assistant_score / self.scale_assistant_score)
        assistant_score = self.scale_assistant_score * assistant_confidence_score

        ep_score = (
                self.weight_op_score * op_score + self.weight_nres_score * nres_score + self.weight_assistant_score * assistant_score
        )
        return (ep_score, op_score, nres_score, assistant_score), n_played, total_ts
