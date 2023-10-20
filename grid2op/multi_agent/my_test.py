import grid2op
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv

env = grid2op.make("l2rpn_case14_sandbox", test = True)
action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
observation_domains = {
            'agent_0' : action_domains['agent_1'],
            'agent_1' : action_domains['agent_0']
        }

# run redispatch agent on one scenario for 100 timesteps
ma_env = MultiAgentEnv(env, observation_domains, action_domains)

