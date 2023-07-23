# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest
import os
import tempfile
from grid2op.Observation import BaseObservation
from grid2op.tests.helper_path_test import *

from grid2op import make
from grid2op.Reward import _AlertTrustScore
from grid2op.Parameters import Parameters
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner  # TODO
from grid2op.Action import BaseAction, PlayableAction
from grid2op.Agent import BaseAgent
from grid2op.Episode import EpisodeData

from _aux_opponent_for_test_alerts import (_get_steps_attack,
                                           TestOpponent,
                                           TestOpponentMultiLines)

ATTACKED_LINE = "48_50_136"

DEFAULT_PARAMS_TRUSTSCORE = dict(reward_min_no_blackout=-1.0,
                                 reward_min_blackout=-50.0,
                                 reward_max_no_blackout=0.0,
                                 reward_max_blackout=0.0,
                                 reward_end_episode_bonus=0.0,
                                 min_score=-3.0)

#a near copy of _normalisation_fun function when for to a given trustscore parametrization it is not easy to guess the score before hand
# especially when reward_end_episode_bonus is non null for some non blackout cases
def manual_score (cm_reward,cm_reward_min_ep,cm_reward_max_ep,max_score):

    manual_standardized_score= np.round((cm_reward - cm_reward_min_ep) / (cm_reward_max_ep - cm_reward_min_ep + 1e-5), 4)
    manual_score = DEFAULT_PARAMS_TRUSTSCORE["min_score"] + (
                                max_score - DEFAULT_PARAMS_TRUSTSCORE[
                            "min_score"]) * manual_standardized_score
    return manual_score

# Test alertTrustScore when no blackout and when blackout
class TestAlertTrustScoreNoBlackout(unittest.TestCase):
    """test the basic behavior of the assistant alert feature when no blackout occur """

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )

    def test_assistant_trust_score_no_blackout_no_attack_no_alert(self) -> None : 
        """ When no blackout and no attack occur, and no alert is raised we expect a maximum score
            at the end of the episode and cumulated reward equal to the end of episode bonus

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                obs, score, done, info = env.step(env.action_space())
                if info["opponent_attack_line"] is None : 
                    if i == env.max_episode_duration()-1:
                        total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks= env._reward_helper.template_reward.nb_last_attacks
                        assert total_nb_attacks==0
                        assert nb_last_attacks==0
                        assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]

                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)
                        assert cm_reward_min_ep == 0.
                        assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]

                        assert score == env._reward_helper.template_reward.max_score
                    else : 
                        assert score == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done
    
    def test_assistant_trust_score_no_blackout_no_attack_alert(self) -> None : 
        """ When an alert is raised while no attack / nor blackout occur, we expect a maximum score
            at the end of the episode and cumulated reward equal to the end of episode bonus

        Raises:
            Grid2OpException: raise an exception if an attack occur
        """
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            attackable_line_id=0
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1

                if info["opponent_attack_line"] is None : 
                    if step == env.max_episode_duration():
                        total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks
                        assert total_nb_attacks == 0
                        assert nb_last_attacks == 0

                        assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]
                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)
                        assert cm_reward_min_ep == 0.
                        assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]

                        assert score == env._reward_helper.template_reward.max_score
                    else : 
                        assert score == 0
                else : 
                    raise Grid2OpException('No attack expected')
            
            assert done
# If attack 
    def test_assistant_trust_score_no_blackout_attack_no_alert(self) -> None :
        """ When we don't raise an alert for an attack (at step 1)
            and no blackout occur, we expect a maximum score
            at the end of the episode, a cumulated reward equal to reward_max_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps

        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[1])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnbana"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done: 
                        assert np.round(score,3) == env._reward_helper.template_reward.max_score
                        total_nb_attacks=env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                        assert nb_last_attacks == 0
                        assert total_nb_attacks==1
                        assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]
                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)
                        assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                        assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]
                else :
                    assert score == 0

    def test_assistant_trust_score_no_blackout_attack_alert(self) -> None :
        """When we raise an alert for an attack (at step 1)
            and no blackout occur, we expect a minimum score
            at the end of the episode if end of episode bonus is null (or above otherwise), a cumulated reward equal to reward_min_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnba"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if i == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:

                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 1

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]+ DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]+DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]

                    if (DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] == 0):
                        assert score == DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                    else:
                        assert score > DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                        assert score == manual_score (cm_reward,cm_reward_min_ep,cm_reward_max_ep,env._reward_helper.template_reward.max_score)

                else : 
                    assert score == 0

    def test_assistant_trust_score_no_blackout_attack_alert_too_late(self) -> None :
        """ When we raise an alert too late for an attack (at step 2) but no blackout occur, 
            we expect a maximum score at the end of the episode,
            a cumulated reward equal to reward_max_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps


        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tatsnbaatl"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if step == 2 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:
                    assert score == env._reward_helper.template_reward.max_score
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 1

                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] +\
                            DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]+DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]
                else : 
                    assert score == 0

    def test_assistant_trust_score_no_blackout_attack_alert_too_early(self)-> None :
        """ When we raise an alert too early for an attack (at step 2)
            we expect a maximum score at the end of the episode,
            a cumulated reward equal to reward_max_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps


        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tatsnbaate"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if step == 0 :
                    # An alert is raised at step 0
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:
                    assert score == env._reward_helper.template_reward.max_score
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 1

                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] +\
                            DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]

                else : 
                    assert score == 0

    # 2 ligne attaquées 
    def test_assistant_trust_score_no_blackout_2_attack_same_time_no_alert(self) -> None :
        """ When we don't raise an alert for 2 attacks at the same time (step 1) (considered as a single attack event)
            but no blackout occur, we expect a maximum score
            at the end of the episode, a cumulated reward equal to reward_max_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps

        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[1])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnb2astna"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 1 #1 because to simultaneaous attacks is considered as a signgle attack event

                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] +\
                            DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]*total_nb_attacks
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]*total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]*total_nb_attacks

                    assert score == env._reward_helper.template_reward.max_score
                else : 
                    assert score == 0
    
    def test_assistant_trust_score_no_blackout_2_attack_same_time_1_alert(self) -> None :
        """ When we raise only 1 alert for 2 attacks at the same time (step 2) (considered as a single attack event)
            but no blackout occur, we expect a mean score
            at the end of the episode if no end of episode bonus,
            a cumulated reward equal to (reward_max_no_blackout + reward_min_no_blackout)/2 end of episode bonus.
            score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tatsnb2ast1a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 1 #1 because to simultaneaous attacks is considered as a signgle attack event

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert env._reward_helper.template_reward.cumulated_reward==(DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"])/2 + DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] * total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"] * total_nb_attacks

                    max_score=env._reward_helper.template_reward.max_score
                    mean_score=(max_score + DEFAULT_PARAMS_TRUSTSCORE["min_score"]) / 2
                    if(DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]==0):
                        assert score == mean_score
                    else:
                        assert score > mean_score #assuming reward_end_episode_bonus is always positive of course
                        assert score == manual_score (cm_reward,cm_reward_min_ep,cm_reward_max_ep,max_score)
                else : 
                    assert score == 0

    def test_assistant_trust_score_no_blackout_2_attack_same_time_2_alert(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time (step 2) (considered as a single attack event)
            but no blackout occur, we expect a minimum score
            at the end of the episode if no end of episode bonus,
            a cumulated reward equal to reward_min_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tatsnb2ast2a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_ids = [0, 1]
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": attackable_line_ids})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:

                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 1 #1 because to simultaneaous attacks is considered as a signgle attack event

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] * total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"] * total_nb_attacks

                    if(DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]==0):
                        assert score == DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                    else:
                        assert score > DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                        assert score == manual_score (cm_reward,cm_reward_min_ep,cm_reward_max_ep,env._reward_helper.template_reward.max_score)
                else : 
                    assert score == 0


    def test_assistant_trust_score_no_blackout_2_attack_diff_time_no_alert(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time (step 2)
            but no blackout occur, we expect a maximum score at the end of the episode,
            a cumulated reward equal to 2*reward_max_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1, 1], 
                               steps_attack=[1, 2])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnb2dtna"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                obs, score, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent, multi=True) : 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:
                    assert score == env._reward_helper.template_reward.max_score
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 2

                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] +\
                            total_nb_attacks*DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] * total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"] * total_nb_attacks
                else : 
                    assert score == 0
        
    def test_assistant_trust_score_no_blackout_2_attack_diff_time_2_alert(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time (step 2)
            but no blackout occur, we expect a minimum score at the end of the episode if no bonus,
            a cumulated reward equal to 2*reward_min_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnb2dt2a"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [0]})
                elif step == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:

                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 2

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert env._reward_helper.template_reward.cumulated_reward == DEFAULT_PARAMS_TRUSTSCORE[
                        "reward_end_episode_bonus"] + \
                           total_nb_attacks * DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] * total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"] * total_nb_attacks

                    if(DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]==0):
                        assert score == DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                    else:
                        assert score > DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                        assert score == manual_score (cm_reward,cm_reward_min_ep,cm_reward_max_ep,env._reward_helper.template_reward.max_score)
                else : 
                    assert score == 0

    def test_assistant_trust_score_no_blackout_2_attack_diff_time_alert_first_attack(self) -> None :
        """ When we raise 2 alerts for 2 attacks at the same time (step 2)
            but no blackout occur, we expect a mean score at the end of the episode if no bonus,
            a cumulated reward equal to reward_max_no_blackout + reward_min_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps
        """

        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnb2dtafa"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if step == 1 :
                    act = env.action_space({"raise_alert": [0]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:

                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 2

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert env._reward_helper.template_reward.cumulated_reward == DEFAULT_PARAMS_TRUSTSCORE[
                        "reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"]+DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"]

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] * total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"] * total_nb_attacks

                    max_score=env._reward_helper.template_reward.max_score
                    mean_score=(max_score + DEFAULT_PARAMS_TRUSTSCORE["min_score"]) / 2
                    if(DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]==0):
                        assert score == mean_score
                    else:
                        assert score > mean_score #assuming reward_end_episode_bonus is always positive of course
                        assert score == manual_score (cm_reward,cm_reward_min_ep,cm_reward_max_ep,max_score)
                else : 
                    assert score == 0


    def test_assistant_trust_score_no_blackout_2_attack_diff_time_alert_second_attack(self) -> None :
        """ When we raise 1 alert on the second attack while we have 2 attacks at two times (steps 2 and 3)
            but no blackout occur, we expect a mean score at the end of the episode if no bonus,
            a cumulated reward equal to reward_max_no_blackout + reward_min_no_blackout + end of episode bonus.
            score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=[1,1], 
                                   steps_attack=[2, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsnb2dtasa"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = env.action_space()
                if i == 2 : 
                    act = env.action_space({"raise_alert": [1]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done:
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 0
                    assert total_nb_attacks == 2

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert env._reward_helper.template_reward.cumulated_reward == DEFAULT_PARAMS_TRUSTSCORE[
                        "reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] + DEFAULT_PARAMS_TRUSTSCORE[
                               "reward_max_no_blackout"]

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_min_no_blackout"] * total_nb_attacks
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"] + \
                           DEFAULT_PARAMS_TRUSTSCORE["reward_max_no_blackout"] * total_nb_attacks

                    max_score=env._reward_helper.template_reward.max_score
                    mean_score=(max_score + DEFAULT_PARAMS_TRUSTSCORE["min_score"]) / 2

                    if(DEFAULT_PARAMS_TRUSTSCORE["reward_end_episode_bonus"]==0):
                        assert score == mean_score
                    else:
                        assert score > mean_score
                        assert score == manual_score(cm_reward,cm_reward_min_ep,cm_reward_max_ep,max_score)
                else : 
                    assert score == 0, f"error for step {step}: {score} vs 0"



class TestAlertTrustScoreBlackout(unittest.TestCase):
    """test the basic bahavior of the assistant alert feature when a blackout occur"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )
    
    def get_dn(self, env):
        return env.action_space({})

    def get_blackout(self, env):
        blackout_action = env.action_space({})
        blackout_action.gen_set_bus = [(0, -1)]
        return blackout_action

# Cas avec blackout 1 ligne attaquée
# return -10
    def test_assistant_trust_score_blackout_attack_no_alert(self) -> None :
        """
        When 1 line is attacked at step 3 and we don't raise any alert
        and a blackout occur at step 4
        we expect a score of -10
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent, 
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsbana"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if step == 3 :
                    act = self.get_blackout(env)
                obs, score, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done:
                    assert score == DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 1
                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']# -10
                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']
                    break
                else : 
                    assert score == 0
# return 2
    def test_assistant_trust_score_blackout_attack_raise_good_alert(self) -> None :
        """When 1 line is attacked at step 3 and we raise a good alert
        and a blackout occur at step 4, we expect a maximum score,
        a cumulated reward equal to reward_max_blackout
        score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsbarga"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 2:
                    # I raise the alert (on the right line) just before the opponent attack
                    # opp attack at step = 3, so i = 2
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done:
                    assert score == env._reward_helper.template_reward.max_score
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 1

                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']
                    break
                else : 
                    assert score == 0

# return -10
    def test_assistant_trust_score_blackout_attack_raise_alert_just_before_blackout(self) -> None :
        """
        When 1 line is attacked at step 3 and we raise 1 alert  too late
        and a blackout occur at step 4, we expect a minimum score,
        a cumulated reward equal to reward_min_blackout
        score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                               duration=3, 
                               steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsbarajbb"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    # opponent attack at step 3, so when i = 2
                    # i raise the alert BEFORE that (so when i = 1)
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done: 
                    assert score == DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 1

                    assert env._reward_helper.template_reward.cumulated_reward==DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']
                    break
                else : 
                    assert score == 0
                
    def test_assistant_trust_score_blackout_attack_raise_alert_too_early(self) -> None :
        """
        When 1 line is attacked at step 3 and we raise 1 alert  too early
        and a blackout occur at step 4, we expect a minimum score,
        a cumulated reward equal to reward_min_blackout
        score is otherwise 0 at other time steps
        """
        # return -10
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE], 
                                   duration=3, 
                                   steps_attack=[3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsbarate"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    # opp attacks at step = 3, so i = 2, I raise an alert just before
                    act = env.action_space({"raise_alert": [attackable_line_id]})
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done: 
                    assert score == DEFAULT_PARAMS_TRUSTSCORE["min_score"]
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 1

                    assert env._reward_helper.template_reward.cumulated_reward == DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_min_blackout']

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']
                    break
                else : 
                    assert score == 0

# return 2
    def  test_assistant_trust_score_blackout_2_lines_same_step_in_window_good_alerts(self) -> None :
        """
        When 2 lines are attacked simustaneously at step 2 and we raise 2 alert 
        and a blackout occur at step 4, we expect a maximum score,
        a cumulated reward equal to reward_max_blackout
        score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=3, 
                               steps_attack=[3, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsb2lssiwga"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 2:
                    # attack at step 3, so when i = 2 (which is the right time to send an alert)
                    act = env.action_space({"raise_alert": [0,1]})
                obs, score, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done: 
                    assert score == env._reward_helper.template_reward.max_score
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 1 #1 because to simultaneaous attacks is considered as a signgle attack event

                    assert env._reward_helper.template_reward.cumulated_reward == (DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout']+DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout'])/2#2 here because there are two attacks at the same time, so we take the mean of the individual alert scores

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    #attention, attaque dans une même fenêtre avant blackout ne compte que pour une seule attaque pondérée...
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_min_no_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']

                    break
                else : 
                    assert score == 0

# return -4
    def test_assistant_trust_score_blackout_2_lines_attacked_simulaneous_only_1_alert(self) -> None:
        """
        When 2 lines are attacked simustaneously (considered as a single attack event) at step 2 and we raise only 1 alert
        and a blackout occur at step 4, we expect a mean score,
        a cumulated reward equal to (reward_max_blackout + reward_min_blackout)/2
        score is otherwise 0 at other time steps
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                                   duration=3, 
                                   steps_attack=[3, 3])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponent, 
                  kwargs_opponent=kwargs_opponent,
                  _add_to_name="_tatsb2laso1a"
            ) as env : 
            new_param = Parameters()
            new_param.MAX_LINE_STATUS_CHANGED = 10

            env.change_parameters(new_param)
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                attackable_line_id = 0
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 2:
                    # attack at step 3, so i = 2, which is the 
                    # right time to send an alert
                    act = env.action_space({"raise_alert": [0]})
                obs, score, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done: 
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 1 #1 because to simultaneaous attacks is considered as a signgle attack event

                    cm_reward=env._reward_helper.template_reward.cumulated_reward
                    assert cm_reward == (DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout']+DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_min_blackout'])/2 #2 here because there are two attacks at the same time, so we take the mean of the individual alert scores

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    #attention, attaque dans une même fenêtre avant blackout ne compte que pour une seule attaque pondérée...
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_min_no_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']

                    max_score=env._reward_helper.template_reward.max_score
                    mean_score=(DEFAULT_PARAMS_TRUSTSCORE['min_score']+max_score)/2
                    #assert score > DEFAULT_PARAMS_TRUSTSCORE['min_score']
                    assert score == mean_score
                    #assert score == manual_score(cm_reward,cm_reward_min_ep,cm_reward_max_ep,max_score)
                    break
                else : 
                    assert score == 0

# return 2
    def  test_assistant_trust_score_blackout_2_lines_different_step_in_window_good_alerts(self) -> None : 
        """
        When 2 lines are attacked at different steps 3 and 4 and we raise 2  alert 
        and a blackout occur at step 4, we expect a maximum score,
        a cumulated reward equal to (2*reward_max_blackout)/2
        score is otherwise 0 at other time step
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[3, 4])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsb2ldsiwga"
            ) as env : 
            env.seed(0)
            obs = env.reset()
            step = 0            
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 2 :
                    # opp attack "line 0" at step 3 so i = 2 => good alert
                    act = env.action_space({"raise_alert": [0]})
                elif i == 3 : 
                    # opp attack "line 1" at step 4 so i = 3 => good alert
                    act = env.action_space({"raise_alert": [1]})
                elif i == 4 : 
                    # trigger blackout
                    act = self.get_blackout(env)
                obs, score, done, info = env.step(act)
                step += 1
                
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done : 
                    assert score == env._reward_helper.template_reward.max_score
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 2
                    assert total_nb_attacks == 2

                    cm_reward = env._reward_helper.template_reward.cumulated_reward
                    assert cm_reward == (DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout'] +DEFAULT_PARAMS_TRUSTSCORE[
                                                                                       'reward_max_blackout']) / total_nb_attacks

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    # attention, attaque dans une même fenêtre avant blackout ne compte que pour une seule attaque pondérée...
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_min_blackout']  # +DEFAULT_PARAMS_TRUSTSCORE['reward_min_no_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout']  # +DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']

                    max_score=env._reward_helper.template_reward.max_score
                    assert score == max_score
                    break
                else : 
                    assert score == 0, f"error for step {step}: {score} vs 0"

    def test_assistant_trust_score_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_first_attacked_line(self) -> None:
        """
        When 2 lines are attacked at different steps 3 and 4 and we raise 1 alert on the first attack
        and a blackout occur at step 4, we expect a mean score,
        a cumulated reward equal to (reward_max_blackout + reward_min_blackout)/2
        score is otherwise 0 at other time step
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[3, 4])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsb2ladsiwo1aofal"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):  
                act = self.get_dn(env)
                if i == 2 :
                    # opp attack "line 0" at step 3 so i = 2 => good alert
                    act = env.action_space({"raise_alert": [0]})
                elif i == 3 : 
                    act = self.get_blackout(env)
                obs, score, done, info = env.step(act)
                step += 1  # i = step - 1 at this stage
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done :
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 2
                    assert total_nb_attacks == 2

                    cm_reward = env._reward_helper.template_reward.cumulated_reward
                    assert cm_reward == (DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout']+DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_min_blackout'])/total_nb_attacks

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    #attention, attaque dans une même fenêtre avant blackout ne compte que pour une seule attaque pondérée...
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_min_no_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']

                    mean_score=(DEFAULT_PARAMS_TRUSTSCORE['min_score']+env._reward_helper.template_reward.max_score)/2
                    #assert score > DEFAULT_PARAMS_TRUSTSCORE['min_score']
                    assert score == mean_score
                    #assert score == manual_score(cm_reward,cm_reward_min_ep,cm_reward_max_ep,env._reward_helper.template_reward.max_score)

                    break
                else : 
                    assert score == 0, f"error for step {step}: {score} vs 0"

# return -4
    def test_assistant_trust_score_blackout_2_lines_attacked_different_step_in_window_only_1_alert_on_second_attacked_line(self) -> None:
        """
        When 2 lines are attacked at different steps 2 and 3 and we raise 1 alert on the second attack
        and a blackout occur at step 4, we expect a mean score,
        a cumulated reward equal to (reward_max_blackout + reward_min_blackout)/2
        score is otherwise 0 at other time step
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1,1], 
                               steps_attack=[3, 4])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsb2ladsiwo1aosal"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 :
                    # opp attack "line 1" at step 4 so i = 3 => good alert
                    act = env.action_space({"raise_alert": [1]})
                elif i == 4 : 
                    act = self.get_blackout(env)
                obs, score, done, info = env.step(act)
                step += 1
                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                    
                if done :
                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 2
                    assert total_nb_attacks == 2

                    cm_reward = env._reward_helper.template_reward.cumulated_reward
                    assert cm_reward == (DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout']+DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_min_blackout'])/total_nb_attacks

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)
                    #attention, attaque dans une même fenêtre avant blackout ne compte que pour une seule attaque pondérée...
                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_min_no_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']

                    mean_score=(DEFAULT_PARAMS_TRUSTSCORE['min_score']+env._reward_helper.template_reward.max_score)/2
                    assert score == mean_score
                    break
                else : 
                    assert score == 0, f"error for step {step}: {score} vs 0"

# return 2 
    def test_assistant_trust_score_blackout_2_lines_attacked_different_1_in_window_1_good_alert(self) -> None:
        """
        When 2 lines are attacked at different steps 3 and 6 and we raise 1 alert at step 5 on the second attack
        and a blackout occur at step 6, we expect a maximum score,
        a cumulated reward equal to reward_max_blackout + reward_max_no_blackout
        score is otherwise 0 at other time step
        """
        kwargs_opponent = dict(lines_attacked=[ATTACKED_LINE]+['48_53_141'], 
                               duration=[1, 1], 
                               steps_attack=[3, 6])
        with make(self.env_nm,
                  test=True,
                  difficulty="1", 
                  opponent_attack_cooldown=0, 
                  opponent_attack_duration=99999, 
                  opponent_budget_per_ts=1000, 
                  opponent_init_budget=10000., 
                  opponent_action_class=PlayableAction, 
                  opponent_class=TestOpponentMultiLines, 
                  kwargs_opponent=kwargs_opponent,
                  reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE),
                  _add_to_name="_tatsb2lad1iw1ga"
            ) as env : 
            env.seed(0)
            env.reset()
            step = 0
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 5 :
                    # opp attack "line 1" at step 6 so i = 5 => good alert
                    act = env.action_space({"raise_alert": [1]})
                elif i == 6 : 
                    act = self.get_blackout(env)
                obs, score, done, info = env.step(act)
                step += 1

                if step in _get_steps_attack(kwargs_opponent, multi=True): 
                    assert info["opponent_attack_line"] is not None, f"no attack is detected at step {step}"
                else:
                    assert info["opponent_attack_line"]  is None, f"an attack is detected at step {step}"
                
                if done : 
                    assert score == env._reward_helper.template_reward.max_score
                    assert done

                    total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                    nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                    assert nb_last_attacks == 1
                    assert total_nb_attacks == 2

                    assert env._reward_helper.template_reward.cumulated_reward == (DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_blackout']+DEFAULT_PARAMS_TRUSTSCORE[
                        'reward_max_no_blackout'])

                    cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                        total_nb_attacks,nb_last_attacks)

                    assert cm_reward_min_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_min_blackout']+DEFAULT_PARAMS_TRUSTSCORE['reward_min_no_blackout']
                    assert cm_reward_max_ep == DEFAULT_PARAMS_TRUSTSCORE['reward_max_blackout']+DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']#+DEFAULT_PARAMS_TRUSTSCORE['reward_max_no_blackout']
                    break
                else : 
                    assert score == 0, f"error for step {step}: {score} vs 0"

# return 0 
    def test_assistant_trust_score_blackout_no_attack_alert(self) -> None :

        """Even if there is a blackout, an we raise an alert
           we expect a score of 0 because there is no attack and we don't finish the scenario"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 1:
                    act = env.action_space({"raise_alert": [0]})
                obs, score, done, info = env.step(act)
                if info["opponent_attack_line"] is None:
                    if done : #info["opponent_attack_line"] is None :
                        assert score == 0.
                        total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                        assert nb_last_attacks == 0
                        assert total_nb_attacks == 0

                        assert env._reward_helper.template_reward.total_nb_attacks==0.
                        assert env._reward_helper.template_reward.cumulated_reward==0.

                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)

                        assert cm_reward_min_ep == 0.0
                        assert cm_reward_max_ep == 0.0
                else : 
                    raise Grid2OpException('No attack expected')

                if done :
                    break
            
            assert done

# return 0 
    def test_assistant_trust_score_blackout_no_attack_no_alert(self) -> None :
        """Even if there is a blackout, an we don't raise an alert
           we expect a score of 0 because there is no attack and we don't finish the scenario"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                obs, score, done, info = env.step(act)
                if info["opponent_attack_line"] is None :
                    if done:
                        assert score == 0.
                        total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                        assert nb_last_attacks == 0
                        assert total_nb_attacks == 0

                        assert env._reward_helper.template_reward.total_nb_attacks == 0.
                        assert env._reward_helper.template_reward.cumulated_reward == 0.

                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)

                        assert cm_reward_min_ep == 0.0
                        assert cm_reward_max_ep == 0.0
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_trust_score_blackout_no_attack_before_window_alert(self) -> None :
        """Even if there is a blackout, an we raise an alert too early
           we expect a score of 0 because there is no attack and we don't finish the scenario"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i in [0, 1, 2]:
                    act = env.action_space({"raise_alert": [0]})
                obs, score, done, info = env.step(act)
                if info["opponent_attack_line"] is None : 
                    assert score == 0.
                    if done:
                        total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                        assert nb_last_attacks == 0
                        assert total_nb_attacks == 0

                        assert env._reward_helper.template_reward.total_nb_attacks == 0.
                        assert env._reward_helper.template_reward.cumulated_reward == 0.

                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)

                        assert cm_reward_min_ep == 0.0
                        assert cm_reward_max_ep == 0.0
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done

# return 0 
    def test_assistant_trust_score_blackout_no_attack_before_window_no_alert(self) -> None :
        """Even if there is a blackout, an we raise an alert too late
           we expect a score of 0 because there is no attack and we don't finish the scenario"""
        with make(
            self.env_nm,
            test=True,
            difficulty="1",
            reward_class=_AlertTrustScore(**DEFAULT_PARAMS_TRUSTSCORE)
        ) as env:
            env.seed(0)
            env.reset()

            done = False
            for i in range(env.max_episode_duration()):
                act = self.get_dn(env)
                if i == 3 : 
                    act = self.get_blackout(env)
                elif i == 4:
                    # we never go here ...
                    act = env.action_space({"raise_alert": [0]})
                obs, score, done, info = env.step(act)
                
                if info["opponent_attack_line"] is None : 
                    assert score == 0.
                    if done:

                        total_nb_attacks = env._reward_helper.template_reward.total_nb_attacks
                        nb_last_attacks = env._reward_helper.template_reward.nb_last_attacks

                        assert nb_last_attacks == 0
                        assert total_nb_attacks == 0

                        assert env._reward_helper.template_reward.total_nb_attacks == 0.
                        assert env._reward_helper.template_reward.cumulated_reward == 0.

                        cm_reward_min_ep, cm_reward_max_ep = env._reward_helper.template_reward._compute_min_max_reward(
                            total_nb_attacks,nb_last_attacks)

                        assert cm_reward_min_ep == 0.0
                        assert cm_reward_max_ep == 0.0
                else : 
                    raise Grid2OpException('No attack expected')

                if done : 
                    break
            
            assert done


if __name__ == "__main__":
    unittest.main()
