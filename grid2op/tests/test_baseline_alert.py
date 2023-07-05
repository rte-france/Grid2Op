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

            res = runner.run(nb_episode=1, nb_process=1, path_save=None,agent_seeds=[0],env_seeds=[0],max_iter=3,
                             add_detailed_output=True)
            id_chron, name_chron, cum_reward, nb_time_step, max_ts, episode_data = res[0]

            # test if the number of alerts sent on lines are recovered
            alerts_count =np.sum([obs.active_alert for obs in episode_data.observations[1:]]
                                  ,axis=0)
            print(alerts_count)
            assert(np.all(alerts_count==[3, 3, 2, 0, 1, 0, 0, 0, 0, 0]))

            # test that we observe the expected alert rate
            nb_alertable_lines =len(env.alertable_line_names)
            ratio_alerts_step =np.sum(alerts_count ) /(nb_time_step*nb_alertable_lines)
            assert(np.round(ratio_alerts_step ,decimals=1 )==np.round(percentage_alert/100 ,decimals=1))

            #check that alert agent is not doing any intervention on the grid in this short time frame
            #as the reco power line, it should only do actions to reconnect lines when allowed, but cannot happen in this short time frame
            has_action_impact=[act.impact_on_objects()['has_impact'] for act in episode_data.actions]
            assert(~np.any(has_action_impact))