import numpy as np

from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from DoubleDuelingDQN import DoubleDuelingDQN

class DoubleDuelingDQNAgent(AgentWithConverter):
    # first change: An Agent must derived from grid2op.Agent
    # (in this case MLAgent, because we manipulate vector instead of classes)
    
    def convert_obs(self, observation):
        return observation.to_vect()
        
    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)

        self.frames.append(transformed_observation)
        self.frames.pop(0)
        a, _ = self.deep_q.predict_move(np.array(self.frames))
        return a
    
    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            self.num_action = self.action_space.n
            self.deep_q = DoubleDuelingDQN(self.action_space.size(), self.num_action,
                                           transformed_observation.shape[0],
                                           num_frames=self.num_frames,
                                           learning_rate = self.lr)
            self.frames = [transformed_observation.copy() for i in range(self.num_frames)]

    def __init__(self, action_space, num_frames=1, lr=1e-5):
        # Call super constructor
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        self.deep_q = None
        self.num_frames = num_frames
        self.lr=lr
        self.frames = []
    
    def load_network(self, path):
        self.deep_q.load_network(path)
