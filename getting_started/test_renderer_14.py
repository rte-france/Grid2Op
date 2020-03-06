import grid2op
from grid2op.Agent import DoNothingAgent
from grid2op.Agent import GreedyAgent, RandomAgent
import numpy as np
import pdb

env = grid2op.make("case14_realistic")

class MyExpertAgent(GreedyAgent):
    def __init__(self, action_space):
        GreedyAgent.__init__(self, action_space)
        self.saved_score = []
        
    def act(self, observation, reward, done=False):
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        # print("________________\nbeginning simulate")
        self.tested_action = self._get_tested_action(observation)
        if len(self.tested_action) > 1:
            all_rewards = np.full(shape=len(self.tested_action), fill_value=np.NaN, dtype=np.float)
            for i, action in enumerate(self.tested_action):
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                all_rewards[i] = simul_reward
                # if simul_reward > 19:
                #    pdb.set_trace()

            reward_idx = np.argmax(all_rewards)  # rewards.index(max(rewards))
            expected_reward = np.max(all_rewards)
            best_action = self.tested_action[reward_idx]
#             print("Action taken:\n{}".format(best_action))
        else:
            all_rewards = [None]
            expected_reward = None
            best_action = self.tested_action[0]
            
        self.saved_score.append(((best_action, expected_reward),
                                 [el for el in zip(self.tested_action, all_rewards)]))
        # print("end simulate\n_____________")
        return best_action
    
    def _get_tested_action(self, observation):
        res = [self.action_space({})]  # add the do nothing
        for i, el in enumerate(observation.line_status):
            # try to reconnect powerlines
            if not el:
                tmp = np.zeros(self.action_space.n_line, dtype=np.int)
                tmp[i] = 1
                action = self.action_space({"set_line_status": tmp})
                action = action.update({"set_bus": {"lines_or_id": [(i, 1)], "lines_ex_id": [(i, 1)]}})
                res.append(action)
                
        # disconnect the powerlines
        ## 12 to 13, 10 to 9 # 5 to 12, 5 to 10, 
        for i in [19, 17]: # , 10 ,12 <- with that it takes action that leads to divergence, check that!
            tmp = np.full(self.action_space.n_line, fill_value=False, dtype=np.bool)
            tmp[i] = True
            action = self.action_space({"change_line_status": tmp})
            if not observation.line_status[i]:
                # so the action consisted in reconnecting the powerline
                # i need to say on which bus
                action = action.update({"set_bus": {"lines_or_id": [(i, 1)], "lines_ex_id": [(i, 1)]}})
            res.append(action)
        
        # play with the topology
        ## i put powerlines going from 1 to 4 with powerline going from 3 to 4 at substation 4
        action = self.action_space({"change_bus":
                                    {"substations_id": [(4, np.array([False, True, True, False, False]))]}})
        res.append(action)
        
        ## i put powerline from 5 to 12 with powerline from 5 to 10 at substation 5
        action = self.action_space({"change_bus":
                                    {"substations_id": [(5, np.array([False, True, False, True, False, False]))]}})
        res.append(action)
        
        ## i put powerline from 1 to 4 with powerline from 1 to 3 with at substation 1
        action = self.action_space({"change_bus":
                                    {"substations_id": [(1, np.array([False, False, True, True, False, False]))]}})
        res.append(action)
        return res
    
my_agent = MyExpertAgent(env.action_space)
# my_agent = RandomAgent(env.action_space)
print("Total unitary action possible: {}".format(my_agent.action_space.n))
      
all_obs = []
obs = env.reset()
all_obs.append(obs)
reward = env.reward_range[0]
done = False
nb_step = 0
# graph_layout = [(280, -81), (100, -270), (-366, -270), (-366, -54), (64, -54), (64, 54), (-450, 0),
#                 (-550, 0), (-326, 54), (-222, 108), (-79, 162), (170, 270), (64, 270), (-222, 216)]
# env.attach_renderer(graph_layout)
while True:
    env.render()
    action = my_agent.act(obs, reward, done)
    obs, reward, done, _ = env.step(action)
    print("Rendering timestep {}".format(nb_step))
    if done:
        break
    all_obs.append(obs)
    nb_step += 1
