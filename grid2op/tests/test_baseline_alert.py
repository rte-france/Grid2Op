# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest 
import numpy as np


from grid2op import make
from grid2op.Runner import Runner
from grid2op.Agent.alertAgent import AlertAgent

# test alert agent no blackout
class TestAlertNoBlackout(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = "l2rpn_idf_2023"

    def test_alert_Agent(self) -> None:
        pct_alerts = [100./23., 100./21., 300./21., 30., 50., 80.]
        ref_alert_counts = {pct_alerts[0]: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            pct_alerts[1]: [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            pct_alerts[2]: [0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            pct_alerts[3]: [0, 0, 1, 0, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            pct_alerts[4]: [1, 2, 2, 0, 3, 3, 3, 3, 3, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            pct_alerts[5]: [1, 2, 2, 0, 3, 3, 3, 3, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                            }
        # 0 in the first (no budget)
        # one per step max in the second (budget for only 1)
        # 3 per step max in the third (budget for only 1) with the one from the first present
        with make(
                self.env_nm,
                test=True,
                difficulty="1"
        ) as env:
            for percentage_alert in pct_alerts:
                env.seed(0)
                env.reset()
                my_agent = AlertAgent(env.action_space, percentage_alert=percentage_alert)
                runner = Runner(**env.get_params_for_runner(), agentClass=None ,agentInstance=my_agent)

                res = runner.run(nb_episode=1, nb_process=1, path_save=None, agent_seeds=[0],env_seeds=[0], max_iter=3,
                                add_detailed_output=True)
                id_chron, name_chron, cum_reward, nb_time_step, max_ts, episode_data = res[0]

                # test if the number of alerts sent on lines are recovered
                alerts_count = np.sum([obs.active_alert for obs in episode_data.observations[1:]]
                                    ,axis=0)
                assert(np.all(alerts_count == ref_alert_counts[percentage_alert])), f"for {percentage_alert} : {alerts_count} vs {ref_alert_counts[percentage_alert]}"

                # test that we observe the expected alert rate
                nb_alertable_lines =len(env.alertable_line_names)
                ratio_alerts_step =np.sum(alerts_count ) /(nb_time_step*nb_alertable_lines)
                assert(np.round(ratio_alerts_step, decimals=1) <= np.round(percentage_alert/100. ,decimals=1))

                #check that alert agent is not doing any intervention on the grid in this short time frame
                #as the reco power line, it should only do actions to reconnect lines when allowed, but cannot happen in this short time frame
                has_action_impact=[act.impact_on_objects()['has_impact'] for act in episode_data.actions]
                assert(~np.any(has_action_impact))
                
            
if __name__ == "__main__":
    unittest.main()
