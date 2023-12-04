# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.utils.l2rpn_2020_scores import ScoreL2RPN2020
from grid2op.Reward import L2RPNWCCI2022ScoreFun


class ScoreL2RPN2022(ScoreL2RPN2020):
    """This class implements the score used for the L2RPN 2022 competition, 
    taking place in the context of the WCCI 2022 competition.
    """
    def __init__(self,
                 env,
                 env_seeds=None,
                 agent_seeds=None,
                 nb_scenario=16,
                 min_losses_ratio=0.8,
                 verbose=0, max_step=-1,
                 nb_process_stats=1,
                 scores_func=L2RPNWCCI2022ScoreFun,
                 score_names=None,
                 add_nb_highres_sim=False):
        super().__init__(env,
                         env_seeds,
                         agent_seeds,
                         nb_scenario,
                         min_losses_ratio,
                         verbose,
                         max_step,
                         nb_process_stats,
                         scores_func,
                         score_names,
                         add_nb_highres_sim=add_nb_highres_sim)
