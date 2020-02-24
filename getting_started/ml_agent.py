# Credit Abhinav Sagar: 
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial
# Code under MIT license, available at:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial/blob/master/LICENSE
from collections import deque
import random
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import keras
    import keras.backend as K
    from keras.models import load_model, Sequential, Model
    from keras.optimizers import Adam
    from keras.layers.core import Activation, Dropout, Flatten, Dense
    from keras.layers import Input, Lambda
    
from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converters import IdToAct
from grid2op.Reward import RedispReward

class TrainingParam(object):
    def __init__(self,
                    DECAY_RATE = 0.99,
                    BUFFER_SIZE = 40000,
                    MINIBATCH_SIZE = 64,
                    TOT_FRAME = 3000000,
                    EPSILON_DECAY = 1000000,
                    MIN_OBSERVATION = 50, #5000
                    FINAL_EPSILON = 1/300,  # have on average 1 random action per scenario of approx 287 time steps
                    INITIAL_EPSILON = 0.1,
                    TAU = 0.01,
                NUM_FRAMES=1):
        self.DECAY_RATE = DECAY_RATE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.TOT_FRAME = TOT_FRAME
        self.EPSILON_DECAY = EPSILON_DECAY
        self.MIN_OBSERVATION = MIN_OBSERVATION #5000
        self.FINAL_EPSILON = FINAL_EPSILON  # have on average 1 random action per scenario of approx 287 time steps
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.TAU = TAU
        self.NUM_FRAMES = NUM_FRAMES
    
class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        """Add an experience to the buffer"""
        # S represents current state, a is action,
        # r is reward, d is whether it is the end, 
        # and s2 is next state
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
        
# Credit Abhinav Sagar: 
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial
# Code under MIT license, available at:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial/blob/master/LICENSE

class DeepQ(TrainingParam):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size, observation_size, lr=1e-5, **kwargs):
        # It is not modified from  Abhinav Sagar's code, except for adding the possibility to change the learning rate
        # in parameter is also present the size of the action space
        # (it used to be a global variable in the original code)
        TrainingParam.__init__(self, **kwargs)
        self.action_size = action_size
        self.observation_size = observation_size
        self.model = None
        self.target_model = None
        self.lr_ = lr
        self.construct_q_network()
    
    def construct_q_network(self):
        # replacement of the Convolution layers by Dense layers, and change the size of the input space and output space
        
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        self.model.add(Dense(self.observation_size*self.NUM_FRAMES))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.observation_size))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.observation_size))
        self.model.add(Activation('relu'))
        self.model.add(Dense(2*self.action_size))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.action_size))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        # Creates a target network as described in DeepMind paper
        self.target_model = Sequential()
        self.target_model.add(Dense(self.observation_size*self.NUM_FRAMES))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Dense(self.observation_size))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Dense(self.observation_size))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Dense(2*self.action_size))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Dense(self.action_size))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.target_model.set_weights(self.model.get_weights())
    
    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # nothing has changed from the original implementation
        rand_val = np.random.random()
        q_actions = self.model.predict(data.reshape(1, self.observation_size*self.NUM_FRAMES), batch_size = 1)
        
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.action_size)
        else:
            opt_policy = np.argmax(np.abs(q_actions))
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        # nothing has changed from the original implementation, except for changing the input dimension 'reshape'
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, self.observation_size*self.NUM_FRAMES), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, self.observation_size*self.NUM_FRAMES), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += self.DECAY_RATE * np.max(fut_action)
        loss = self.model.train_on_batch(s_batch, targets)
        # Print the loss every 100 iterations.
        if observation_num % 100 == 0:
            print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        # nothing has changed
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.TAU * model_weights[i] + (1 - self.TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)
        
class DuelQ(TrainingParam):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size,  observation_size, lr=0.00001, **kwargs):
        # It is not modified from  Abhinav Sagar's code, except for adding the possibility to change the learning rate
        # in parameter is also present the size of the action space
        # (it used to be a global variable in the original code)
        TrainingParam.__init__(self, **kwargs)
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_ = lr
        self.model = None
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self.model = Sequential()
        
        input_layer = Input(shape = (self.observation_size*self.NUM_FRAMES,))
        lay1 = Dense(self.observation_size*self.NUM_FRAMES)(input_layer)
        lay1 = Activation('relu')(lay1)
        
        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)
        
        lay3 = Dense(2*self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)
        
        fc1 = Dense(self.action_size)(lay3)
        advantage = Dense(self.action_size)(fc1)
        fc2 = Dense(self.action_size)(lay3)
        value = Dense(1)(fc2)
        
        meaner = Lambda(lambda x: K.mean(x, axis=1) )
        mn_ = meaner(advantage)  
        tmp = keras.layers.subtract([advantage, mn_])  # keras doesn't like this part...
        policy = keras.layers.add([tmp, value])

        self.model = Model(inputs=[input_layer], outputs=[policy])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        self.target_model = Model(inputs=[input_layer], outputs=[policy])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        print("Successfully constructed networks.")
    
    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # only changes lie in adapting the input shape
        q_actions = self.model.predict(data.reshape(1, self.observation_size*self.NUM_FRAMES), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        # nothing has changed except adapting the input shapes
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, self.observation_size*self.action_size), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, self.observation_size*self.action_size), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += self.DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 100 iterations.
        if observation_num % 100 == 0:
            print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        # nothing has changed
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)

        
class DeepQAgent(AgentWithConverter):
    # first change: An Agent must derived from grid2op.Agent (in this case MLAgent, because we manipulate vector instead
    # of classes)
    
    def convert_obs(self, observation):
        return observation.to_vect()
        
    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        # print("predict_movement_int: {}".format(predict_movement_int))
        return predict_movement_int
    
    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            # the first time an observation is observed, I set up the neural network with the proper dimensions.
            if self.mode == "DDQN":
                self.deep_q = DeepQ(self.action_space.size(), observation_size=transformed_observation.shape[0])
            elif self.mode == "DQN":
                self.deep_q = DuelQ(self.action_space.size(), observation_size=transformed_observation.shape[0])
                
    def __init__(self, action_space, mode="DDQN", training_param=TrainingParam()):
        # this function has been adapted.
        
        # to built a AgentWithConverter, we need an action_space. 
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)
        self.training_param = training_param 
        # and now back to the origin implementation
        self.replay_buffer = ReplayBuffer(self.training_param.BUFFER_SIZE)
        
        # compare to original implementation, i don't know the observation space size. 
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.deep_q = None
        self.mode = mode
        self.process_buffer = []
    
    def load_network(self, path):
        # not modified compare to original implementation
        self.deep_q.load_network(path)
    
    def convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        # here i simply concatenate the action in case of multiple action in the "buffer"
        # this function existed in the original implementation, bus has been adapted.
        if self.training_param.NUM_FRAMES != 1:
            raise RuntimeError("can only use self.training_param.NUM_FRAMES = 1 for now")
        return np.array(self.process_buffer)
        # TODO fix cases where NUM_FRAMES is not 1 !!!!
        
#         return np.concatenate(self.process_buffer)
    

class TrainAgent(object):
    def __init__(self, agent, reward_fun=RedispReward, env=None):
        self.agent = agent
        self.reward_fun = reward_fun
        self.env = env
        
    def _build_valid_env(self, training_param):
        # now we are creating a valid Environment
        # it's mandatory because no environment are created when the agent is 
        # an Agent should not get direct access to the environment, but can interact with it only by:
        # * receiving reward
        # * receiving observation
        # * sending action
        
        close_env = False
        
        if self.env is None:
            self.env = grid2op.make(action_class=type(self.agent.action_space({})),
                                    reward_class=self.reward_fun)
            close_env = True
                               
        # I make sure the action space of the user and the environment are the same.
        if not isinstance(self.agent.init_action_space, type(self.env.action_space)):
            raise RuntimeError("Imposssible to build an agent with 2 different action space")
        if not isinstance(self.env.action_space, type(self.agent.init_action_space)):
            raise RuntimeError("Imposssible to build an agent with 2 different action space")
        
        # A buffer that keeps the last `NUM_FRAMES` images
        self.agent.replay_buffer.clear()
        self.agent.process_buffer = []
        
        # make sure the environment is reset
        obs = self.env.reset()
        self.agent.process_buffer.append(self.agent.convert_obs(obs))
        do_nothing = self.env.action_space()
        for _ in range(training_param.NUM_FRAMES-1):
            # Initialize buffer with the first frames
            s1, r1, _, _ = self.env.step(do_nothing)
            self.agent.process_buffer.append(self.agent.convert_obs(s1))            
        return close_env
    
    def train(self, num_frames, env=None, training_param=TrainingParam()):
        # this function existed in the original implementation, but has been slightly adapted.
        
        # first we create an environment or make sure the given environment is valid
        close_env = self._build_valid_env(training_param)
        
        # bellow that, only slight modification has been made. They are highlighted
        observation_num = 0
        curr_state = self.agent.convert_process_buffer()
        epsilon = self.training_param.INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" %observation_num))

            # Slowly decay the learning rate
            if epsilon > training_param.FINAL_EPSILON:
                epsilon -= (training_param.INITIAL_EPSILON-training_param.FINAL_EPSILON)/training_param.EPSILON_DECAY

            initial_state = self.agent.convert_process_buffer()
            self.agent.process_buffer = []

            # it's a bit less convenient that using the SpaceInvader environment.
            # first we need to initiliaze the neural network
            self.agent.init_deep_q(curr_state)
            # then we need to predict the next move
            predict_movement_int, predict_q_value = self.agent.deep_q.predict_movement(curr_state, epsilon)
            # and then we convert it to a valid action
            act = self.agent.convert_act(predict_movement_int)
            
            reward, done = 0, False
            for i in range(NUM_FRAMES):
                temp_observation_obj, temp_reward, temp_done, _ = self.env.step(act)
                # here it has been adapted too. The observation get from the environment is
                # first converted to vector
                
                # below this line no changed have been made to the original implementation.
                reward += temp_reward
                self.agent.process_buffer.append(self.agent.convert_obs(temp_observation_obj))
                done = done | temp_done

            if done:
                print("Lived with maximum time ", alive_frame)
                print("Earned a total of reward equal to ", total_reward)
                # reset the environment
                self.env.reset()
                
                alive_frame = 0
                total_reward = 0

            new_state = self.agent.convert_process_buffer()
            self.agent.replay_buffer.add(initial_state, predict_movement_int, reward, done, new_state)
            total_reward += reward
            if self.agent.replay_buffer.size() > training_param.MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.agent.replay_buffer.sample(training_param.MINIBATCH_SIZE)
                self.agent.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.agent.deep_q.target_train()

            # Save the network every 100000 iterations
            if observation_num % 10000 == 9999 or observation_num == num_frames-1:
                print("Saving Network")
                self.agent.deep_q.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1
            
        if close_env:
            print("closing env")
            self.env.close()
            
    def calculate_mean(self, num_episode = 100, env=None):
        # this method has been only slightly adapted from the original implementation
        
        # Note that it is NOT the recommended method to evaluate an Agent. Please use "Grid2Op.Runner" instead
        
        # first we create an environment or make sure the given environment is valid
        close_env = self._build_valid_env()
        
        reward_list = []
        print("Printing scores of each trial")
        for i in range(num_episode):
            done = False
            tot_award = 0
            self.env.reset()
            while not done:
                state = self.convert_process_buffer()
                
                # same adapation as in "train" function. 
                predict_movement_int = self.agent.deep_q.predict_movement(state, 0.0)[0]
                predict_movement = self.agent.convert_act(predict_movement_int)
                
                # same adapation as in the "train" funciton
                observation_obj, reward, done, _ = self.env.step(predict_movement)
                observation_vect_full = observation_obj.to_vect()
                
                tot_award += reward
                self.process_buffer.append(observation)
                self.process_buffer = self.process_buffer[1:]
            print(tot_award)
            reward_list.append(tot_award)
            
        if close_env:
            self.env.close()
        return np.mean(reward_list), np.std(reward_list)