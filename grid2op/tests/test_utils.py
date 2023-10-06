# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.tests.helper_path_test import *
import unittest

PATH_ADN_CHRONICS_FOLDER = os.path.abspath(
    os.path.join(PATH_CHRONICS, "test_multi_chronics")
)
PATH_PREVIOUS_RUNNER = os.path.join(data_test_dir, "runner_data")

from grid2op.Reward import L2RPNSandBoxScore
from grid2op.MakeEnv import make
from grid2op.dtypes import dt_float
from grid2op.Agent import DoNothingAgent, RecoPowerlineAgent
from grid2op.utils import EpisodeStatistics, ScoreL2RPN2020, ScoreICAPS2021
from grid2op.Parameters import Parameters

import warnings


class TestEpisodeStatistics(HelperTests, unittest.TestCase):
    """test teh grid2op.utils.EpisodeStatistics"""

    def test_read(self):
        """test that i can read the data stored"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                obs = env.reset()
                stats = EpisodeStatistics(env)
                aor_, ids_ = stats.get("a_or")
                assert aor_.shape == (7930, 8)
                assert np.max(ids_) == 19
                assert ids_.shape == (7930, 1)
                assert self.compare_vect(
                    np.mean(aor_, axis=0),
                    np.array(
                        [
                            351.6208,
                            153.674,
                            91.057,
                            80.47367,
                            351.93213,
                            89.18627,
                            89.18627,
                            74.77638,
                        ],
                        dtype=dt_float,
                    ),
                )

    def test_compute_erase(self):
        """test that i can compute and erase the results afterwards"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                stats = EpisodeStatistics(env, "test")
                stats.compute(nb_scenario=1, max_step=10, pbar=False)
                # the file have been created
                assert os.path.exists(
                    os.path.join(env.get_path_env(), stats.get_name_dir("test"))
                )
                # i can access it
                aor_, ids_ = stats.get("a_or")
                assert aor_.shape == (11, 8)
                # i can clear the data of individual episode
                stats.clear_episode_data()
                assert not os.path.exists(
                    os.path.join(env.get_path_env(), stats.get_name_dir("test"), "00")
                )
                # i can clear everything
                stats.clear_all()
                assert not os.path.exists(
                    os.path.join(env.get_path_env(), stats.get_name_dir("test"))
                )

    def test_compute_with_score(self):
        """test that i can compute and erase the results afterwards"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                stats = EpisodeStatistics(env, "test")
                stats.compute(
                    nb_scenario=2,
                    max_step=10,
                    pbar=False,
                    scores_func=L2RPNSandBoxScore,
                )
                # i can access it
                scores, ids_ = stats.get(EpisodeStatistics.SCORES)
                assert scores.shape == (20,), "error on the score shape"
                assert ids_.shape == (20, 1), "error on the ids shape"

                scores, ids_ = stats.get("scores")
                assert scores.shape == (20,), "error on the score shape"
                assert ids_.shape == (20, 1), "error on the ids shape"
                # i can clear everything
                stats.clear_all()
                assert not os.path.exists(
                    os.path.join(env.get_path_env(), stats.get_name_dir("test"))
                )

    def test_compute_without_score(self):
        """test that i can compute and erase the results afterwards"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                stats = EpisodeStatistics(env, "test")
                stats.compute(nb_scenario=2, max_step=10, pbar=False)
                # i can access it
                prods, ids_ = stats.get("prod_p")
                assert prods.shape == (22, 2), "error on the prods shape"
                assert ids_.shape == (22, 1), "error on the ids shape"
                with self.assertRaises(RuntimeError):
                    scores, ids_ = stats.get("scores")

                # i can clear everything
                stats.clear_all()
                assert not os.path.exists(
                    os.path.join(env.get_path_env(), stats.get_name_dir("test"))
                )


class TestL2RPNSCORE(HelperTests, unittest.TestCase):
    """test teh grid2op.utils.EpisodeStatistics"""

    def test_can_compute(self):
        """test that i can initialize the score and then delete the statistics"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                scores = ScoreL2RPN2020(env, nb_scenario=4, verbose=0, max_step=50)

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                       EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

    def test_donothing_0(self):
        """test that do nothing has a score of 0.00"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                scores = ScoreL2RPN2020(env, nb_scenario=4, verbose=0, max_step=20)

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                my_agent = DoNothingAgent(env.action_space)
                my_scores, *_ = scores.get(my_agent)
                assert np.max(np.abs(my_scores)) <= self.tol_one

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

    def test_modif_max_step_decrease(self):
        """
        test that i can modify the max step by decreaseing it (and in that case it does not trigger a recomputation
        of the statistics)
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                scores = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=15)

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                my_agent = DoNothingAgent(env.action_space)
                my_scores, *_ = scores.get(my_agent)
                assert (
                    np.max(np.abs(my_scores)) <= self.tol_one
                ), "error for the first do nothing"

                scores2 = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=10)
                assert not scores2._recomputed_dn
                assert not scores2._recomputed_no_ov_rp
                my_agent = DoNothingAgent(env.action_space)
                my_scores2, *_ = scores2.get(my_agent)
                assert (
                    np.max(np.abs(my_scores2)) <= self.tol_one
                ), "error for the second do nothing"

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

    def test_modif_max_step_increase(self):
        """test that i can modify the max step (and that if I increase it it does trigger a recomputation)"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                scores = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=5)

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                my_agent = DoNothingAgent(env.action_space)
                my_scores, *_ = scores.get(my_agent)
                assert (
                    np.max(np.abs(my_scores)) <= self.tol_one
                ), "error for the first do nothing"

                scores2 = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=10)
                assert scores2._recomputed_dn
                assert scores2._recomputed_no_ov_rp

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

    def test_modif_nb_scenario(self):
        """
        test that i can modify the nb_scenario and it properly recomputes it when it increased and not
        when it decreases
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                scores = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=5)

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                my_agent = DoNothingAgent(env.action_space)
                my_scores, *_ = scores.get(my_agent)
                assert (
                    np.max(np.abs(my_scores)) <= self.tol_one
                ), "error for the first do nothing"

                scores2 = ScoreL2RPN2020(env, nb_scenario=4, verbose=0, max_step=5)
                assert scores2._recomputed_dn
                assert scores2._recomputed_no_ov_rp

                scores2 = ScoreL2RPN2020(env, nb_scenario=3, verbose=0, max_step=5)
                assert not scores2._recomputed_dn
                assert not scores2._recomputed_no_ov_rp

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

    def test_reco_noov_80(self):
        """test that do nothing has a score of 80.0 if it is run with "no overflow disconnection" """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make("rte_case5_example", test=True) as env:
                # I cannot decrease the max step: it must be above the number of steps the do nothing does
                scores = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=130)
                assert scores._recomputed_dn
                assert scores._recomputed_no_ov_rp

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                my_agent = DoNothingAgent(env.action_space)
                my_scores, *_ = scores.get(my_agent)
                assert (
                    np.max(np.abs(my_scores)) <= self.tol_one
                ), "error for the first do nothing"

            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
            with make("rte_case5_example", test=True, param=param) as env:
                scores2 = ScoreL2RPN2020(env, nb_scenario=2, verbose=0, max_step=130)
                assert not scores2._recomputed_dn
                assert not scores2._recomputed_no_ov_rp
                my_agent = RecoPowerlineAgent(env.action_space)
                my_scores, *_ = scores2.get(my_agent)
                assert np.max(np.abs(np.array(my_scores) - 80.0)) <= self.tol_one

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )


class TestICAPSSCORE(HelperTests, unittest.TestCase):
    """test teh grid2op.utils.EpisodeStatistics"""

    def test_can_compute(self):
        """test that i can initialize the score and then delete the statistics"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with make(
                os.path.join(PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"),
                test=True,
            ) as env:
                scores = ScoreICAPS2021(
                    env,
                    nb_scenario=2,
                    verbose=0,
                    max_step=50,
                    env_seeds=[1, 2],  # with these seeds do nothing goes till the end
                    agent_seeds=[3, 4],
                )
                my_agent = DoNothingAgent(env.action_space)
                scores_this, n_played, total_ts = scores.get(my_agent)
                for (ep_score, op_score, alarm_score) in scores_this:
                    assert (
                        np.abs(ep_score - 30.0) <= self.tol_one
                    ), f"wrong score for the episode: {ep_score} vs 30."
                    assert np.abs(op_score - 0.0) <= self.tol_one, (
                        f"wrong score for the operationnal cost: " f"{op_score} vs 0."
                    )
                    assert np.abs(alarm_score - 100.0) <= self.tol_one, (
                        f"wrong score for the alarm: " f"{alarm_score} vs 100."
                    )

                # the statistics have been properly computed
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreICAPS2021.NAME_DN),
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreICAPS2021.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )

                # delete them
                scores.clear_all()
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN),
                    )
                )
                # assert not os.path.exists(os.path.join(env.get_path_env(),
                #                                        EpisodeStatistics.get_name_dir(ScoreL2RPN2020.NAME_DN_NO_OVERWLOW)))
                assert not os.path.exists(
                    os.path.join(
                        env.get_path_env(),
                        EpisodeStatistics.get_name_dir(
                            ScoreL2RPN2020.NAME_RP_NO_OVERFLOW
                        ),
                    )
                )


if __name__ == "__main__":
    unittest.main()
