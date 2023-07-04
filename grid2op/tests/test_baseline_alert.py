from grid2op.tests.helper_path_test import *

from grid2op import make
from grid2op.Reward import AlertReward
from grid2op.Runner import Runner
from grid2op.Agent.alertAgent import AlertAgent

# test alert agent no blackout
class TestAlertNoBlackout(unittest.TestCase):
    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_idf_2023_with_alert"
        )

    def test_alert_Agent(self) -> None:
        with make(
                self.env_nm,
                test=True,
                difficulty="1",
                reward_class=AlertReward(reward_end_episode_bonus=42)
        ) as env:
            env.seed(0)
            env.reset()

            percentage_alert =30  # 30% of lines with alert per step
            my_agent = AlertAgent(env.action_space, percentage_alert=percentage_alert)
            runner = Runner(**env.get_params_for_runner(), agentClass=None ,agentInstance=my_agent)
            # runner = Runner(**env.get_params_for_runner(), agentClass=AlertAgent)
            res = runner.run(nb_episode=1, nb_process=1, path_save=None,
                             add_detailed_output=True)

            # test if the number of alerts sent er lines are recovered
            alerts_count =np.sum([res[0][5].observations[i].active_alert for i in range(1, len(res[0][5].observations))]
                                  ,axis=0)
            assert(np.all(alerts_count==[9, 9, 4, 0, 5, 0, 0, 0, 0, 0]))

            # test that we observe the expected alert rate
            nb_steps =res[0][4]
            nb_alertable_lines =len(env.alertable_line_names)
            ratio_alerts_step =np.sum(alerts_count ) /(nb_steps*nb_alertable_lines)
            assert(np.round(ratio_alerts_step ,decimals=1 )==np.round(percentage_alert/100 ,decimals=1))