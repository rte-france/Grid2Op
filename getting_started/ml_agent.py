from collections import deque
import random
import numpy as np

#tf2.0 friendly
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model, Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, subtract, add
    from tensorflow.keras.layers import Input, Lambda, Concatenate

import grid2op
from grid2op.Agent import AgentWithConverter
from grid2op.Converters import IdToAct
from grid2op.Reward import RedispReward


class TrainingParam(object):
    def __init__(self,
                 DECAY_RATE=0.9,
                 BUFFER_SIZE=40000,
                 MINIBATCH_SIZE=64,
                 TOT_FRAME=3000000,
                 EPSILON_DECAY=10000,
                 MIN_OBSERVATION=50, #5000
                 FINAL_EPSILON=1/300,  # have on average 1 random action per scenario of approx 287 time steps
                 INITIAL_EPSILON=0.1,
                 TAU=0.01,
                 ALPHA=1,
                 NUM_FRAMES=1,
    ):
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
        self.ALPHA = ALPHA


# Credit Abhinav Sagar:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial
# Code under MIT license, available at:
# https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial/blob/master/LICENSE
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
class DeepQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size, observation_size,
                 lr=1e-5,
                 training_param=TrainingParam()):
        # It is not modified from  Abhinav Sagar's code, except for adding the possibility to change the learning rate
        # in parameter is also present the size of the action space
        # (it used to be a global variable in the original code)
        self.action_size = action_size
        self.observation_size = observation_size
        self.model = None
        self.target_model = None
        self.lr_ = lr
        self.qvalue_evolution = []
        self.construct_q_network()
        self.training_param = training_param
    
    def construct_q_network(self):
        # replacement of the Convolution layers by Dense layers, and change the size of the input space and output space
        
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        input_layer = Input(shape=(self.observation_size * self.training_param.NUM_FRAMES,))
        layer1 = Dense(self.observation_size * self.training_param.NUM_FRAMES)(input_layer)
        layer1 = Activation('relu')(layer1)
        layer2 = Dense(self.observation_size)(layer1)
        layer2 = Activation('relu')(layer2)
        layer3 = Dense(self.observation_size)(layer2)
        layer3 = Activation('relu')(layer3)
        layer4 = Dense(2 * self.action_size)(layer3)
        layer4 = Activation('relu')(layer4)
        output = Dense(self.action_size)(layer4)

        self.model = Model(inputs=[input_layer], outputs=[output])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        self.target_model = Model(inputs=[input_layer], outputs=[output])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.target_model.set_weights(self.model.get_weights())
    
    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # nothing has changed from the original implementation
        rand_val = np.random.random()
        q_actions = self.model.predict(data.reshape(1, self.observation_size*self.training_param.NUM_FRAMES),
                                       batch_size=1)
        
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.action_size)
        else:
            opt_policy = np.argmax(np.abs(q_actions))
            
        self.qvalue_evolution.append(q_actions[0,opt_policy])

        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        # nothing has changed from the original implementation, except for changing the input dimension 'reshape'
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES),
                                            batch_size=1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES),
                                                   batch_size=1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += self.training_param.DECAY_RATE * np.max(fut_action)
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
            target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)


class DuelQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size, observation_size,
                 lr=0.00001,
                 training_param=TrainingParam()):
        # It is not modified from  Abhinav Sagar's code, except for adding the possibility to change the learning rate
        # in parameter is also present the size of the action space
        # (it used to be a global variable in the original code)
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_ = lr
        self.model = None
        self.target_model = None
        self.qvalue_evolution = []
        self.training_param = training_param

        self.target_model = None

        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self.model = Sequential()
        
        input_layer = Input(shape=(self.observation_size*self.training_param.NUM_FRAMES,))
        lay1 = Dense(self.observation_size*self.training_param.NUM_FRAMES)(input_layer)
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
        tmp = subtract([advantage, mn_])  # keras doesn't like this part...
        policy = add([tmp, value])

        self.model = Model(inputs=[input_layer], outputs=[policy])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        self.target_model = Model(inputs=[input_layer], outputs=[policy])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        print("Successfully constructed networks.")
    
    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # only changes lie in adapting the input shape
        q_actions = self.model.predict(data.reshape(1, self.observation_size*self.training_param.NUM_FRAMES),
                                       batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.action_size)
            
        self.qvalue_evolution.append(q_actions[0,opt_policy])

        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        # nothing has changed except adapting the input shapes
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES),
                                            batch_size=1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES),
                                                   batch_size=1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += self.training_param.DECAY_RATE * np.max(fut_action)

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


class RealQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size, observation_size, lr=1e-5, mean_reg=False,
                 training_param=TrainingParam()):

        self.action_size = action_size
        self.observation_size = observation_size
        self.model = None
        self.target_model = None
        self.model_copy = None

        self.lr_ = lr
        self.mean_reg = mean_reg
        
        self.qvalue_evolution = []
        self.training_param = training_param

        self.construct_q_network()
        
    def construct_q_network(self):

        self.model = Sequential()
        
        input_states = Input(shape=(self.observation_size,))
        input_action = Input(shape=(self.action_size,))
        input_layer = Concatenate()([input_states, input_action])
        
        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)
        
        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)
        
        lay3 = Dense(2*self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)
        
        fc1 = Dense(self.action_size)(lay3)
        advantage = Dense(1, activation = 'linear')(fc1)
        
        if self.mean_reg:
            advantage = Lambda(lambda x : x - K.mean(x))(advantage)
        
        self.model = Model(inputs=[input_states, input_action], outputs=[advantage])
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        
        self.model_copy = Model(inputs=[input_states, input_action], outputs=[advantage])
        self.model_copy.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.model_copy.set_weights(self.model.get_weights())
        
    def predict_movement(self, states, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # nothing has changed from the original implementation
        rand_val = np.random.random()
        q_actions = self.model.predict([np.tile(states.reshape(1, self.observation_size),(self.action_size,1)),
                                        np.eye(self.action_size)]).reshape(1,-1)
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.action_size)
        else:
            opt_policy = np.argmax(np.abs(q_actions))
            
        self.qvalue_evolution.append(q_actions[0, opt_policy])
        
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        # nothing has changed from the original implementation, except for changing the input dimension 'reshape'
        batch_size = s_batch.shape[0]
        targets = np.zeros(batch_size)
        last_action=np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            last_action[i,a_batch[i]] = 1
            q_pre = self.model.predict([s_batch[i].reshape(1, self.observation_size), last_action[i].reshape(1,-1)],
                                       batch_size=1).reshape(1, -1)
            q_fut = self.model_copy.predict([np.tile(s2_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                             np.eye(self.action_size)]).reshape(1,-1)
            fut_action = np.max(q_fut)
            if d_batch[i] == False:
                targets[i] = self.training_param.ALPHA * (r_batch[i] + self.training_param.DECAY_RATE * fut_action - q_pre)
            else:
                targets[i] = self.training_param.ALPHA * (r_batch[i] - q_pre)
        loss = self.model.train_on_batch([s_batch, last_action], targets)
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
        target_model_weights = self.model_copy.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * target_model_weights[i]
        self.model_copy.set_weights(model_weights)


class SAC(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, action_size, observation_size, lr=1e-5,
                 training_param=TrainingParam()):

        self.action_size = action_size
        self.observation_size = observation_size
        self.model = None
        self.target_model = None
        self.lr_ = lr
        self.average_reward = 0
        self.life_spent = 1
        self.qvalue_evolution = []
        self.Is_nan = False
        self.training_param = training_param

        self.construct_q_network()

        self.model_value_target = None
        self.model_Q = None
        self.model_Q2 = None
        self.model_value = None
        self.model_policy = None

    def build_q_NN(self):
        model = Sequential()
        
        input_states = Input(shape = (self.observation_size,))
        input_action = Input(shape = (self.action_size,))
        input_layer = Concatenate()([input_states, input_action])
        
        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)
        
        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)
        
        lay3 = Dense(2*self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)
        
        advantage = Dense(1, activation = 'linear')(lay3)
        
        model = Model(inputs=[input_states, input_action], outputs=[advantage])
        model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        
        return model
    
    def construct_q_network(self):
        # construct double Q networks
        self.model_Q = self.build_q_NN()
        self.model_Q2 = self.build_q_NN()

        # state value function approximation
        self.model_value = Sequential()
        
        input_states = Input(shape = (self.observation_size,))
        
        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)
        
        lay3 = Dense(2*self.action_size)(lay1)
        lay3 = Activation('relu')(lay3)
        
        advantage = Dense(self.action_size, activation = 'relu')(lay3)
        state_value = Dense(1, activation = 'linear')(advantage)
        
        self.model_value = Model(inputs=[input_states], outputs=[state_value])
        self.model_value.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        
        self.model_value_target = Sequential()
        
        input_states = Input(shape = (self.observation_size,))
        
        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)
        
        lay3 = Dense(2*self.action_size)(lay1)
        lay3 = Activation('relu')(lay3)
        
        advantage = Dense(self.action_size, activation = 'relu')(lay3)
        state_value = Dense(1, activation = 'linear')(advantage)
        
        self.model_value_target = Model(inputs=[input_states], outputs=[state_value])
        self.model_value_target.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.model_value_target.set_weights(self.model_value.get_weights())
        # policy function approximation
        
        self.model_policy = Sequential()
        input_states = Input(shape = (self.observation_size,))
        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)
        
        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)
        
        lay3 = Dense(2*self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)
        
        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)
        
        self.model_policy = Model(inputs=[input_states], outputs=[soft_proba])
        self.model_policy.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr_))
        
        print("Successfully constructed networks.")
        
    def predict_movement(self, states, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        # nothing has changed from the original implementation
        p_actions = self.model_policy.predict(states.reshape(1, self.observation_size)).ravel()
        rand_val = np.random.random()
        
        if rand_val < epsilon / 10:
            opt_policy = np.random.randint(0, self.action_size)
        else:
            #opt_policy = np.random.choice(np.arange(self.action_size, dtype=int), size=1, p = p_actions)
            opt_policy = np.argmax(p_actions)
                            
        return np.int(opt_policy), p_actions[opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains networks to fit given parameters"""

        # nothing has changed from the original implementation, except for changing the input dimension 'reshape'
        batch_size = s_batch.shape[0]
        target = np.zeros((batch_size, 1))
        new_proba = np.zeros((batch_size, self.action_size))
        last_action=np.zeros((batch_size, self.action_size))
        
        # training of the action state value networks
        last_action=np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            last_action[i,a_batch[i]] = 1

            v_t = self.model_value_target.predict(s_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES), batch_size = 1)
            
            self.qvalue_evolution.append(v_t[0])
            fut_action = self.model_value_target.predict(s2_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES), batch_size = 1)

            target[i,0] = r_batch[i] + (1 - d_batch[i]) * self.training_param.DECAY_RATE * fut_action

        loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        loss_2 = self.model_Q2.train_on_batch([s_batch, last_action], target) 
            
        # training of the policy
        for i in range(batch_size):
            self.life_spent += 1
            temp =  1 / np.log(self.life_spent) / 2
            new_values = self.model_Q.predict([np.tile(s_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                                      np.eye(self.action_size)]).reshape(1,-1)
            new_values -= np.amax(new_values, axis=-1)
            new_proba[i] = np.exp(new_values / temp) / np.sum(np.exp(new_values / temp), axis=-1)
            
        loss_policy = self.model_policy.train_on_batch(s_batch, new_proba)
    
        # training of the value_function
        value_target = np.zeros(batch_size)
        for i in range(batch_size):
            target_pi = self.model_policy.predict(s_batch[i].reshape(1, self.observation_size*self.training_param.NUM_FRAMES), batch_size = 1)
            action_v1 = self.model_Q.predict([np.tile(s_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                                      np.eye(self.action_size)]).reshape(1,-1)
            action_v2 = self.model_Q2.predict([np.tile(s_batch[i].reshape(1, self.observation_size),(self.action_size,1)),
                                                      np.eye(self.action_size)]).reshape(1,-1)
            value_target[i] = np.fmin(action_v1[0,a_batch[i]], action_v2[0,a_batch[i]]) - np.sum(target_pi * np.log(target_pi + 1e-6))
        
        loss_value = self.model_value.train_on_batch(s_batch, value_target.reshape(-1,1))
        
        self.Is_nan = np.isnan(loss) + np.isnan(loss_2) + np.isnan(loss_policy) + np.isnan(loss_value)
        # Print the loss every 100 iterations.
        if observation_num % 100 == 0:
            print("We had a loss equal to ", loss, loss_2, loss_policy, loss_value)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model_policy.save("policy_"+path)
        self.model_value_target.save("value_"+path)
        print("Successfully saved network.")

    def load_network(self, path):
        # nothing has changed
        self.model_policy = load_model("policy_"+path)
        self.model_value_target = load_model("value_"+path)
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model_value.get_weights() 
        target_model_weights = self.model_value_target.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * target_model_weights[i]
        self.model_value_target.set_weights(model_weights)


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
            if self.mode == "DQN":
                cls = DeepQ
            elif self.mode == "DDQN":
                cls = DuelQ
            elif self.mode == "RealQ":
                cls = RealQ
            elif self.mode == "SAC":
                cls = SAC
            else:
                raise RuntimeError("Unknown neural network named \"{}\"".format(self.mode))
            self.deep_q = cls(self.action_space.size(), observation_size=transformed_observation.shape[0], lr=self.lr)

    def __init__(self, action_space, mode="DDQN", lr=1e-5, training_param=TrainingParam()):
        # this function has been adapted.

        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        self.replay_buffer = ReplayBuffer(training_param.BUFFER_SIZE)

        # compare to original implementation, i don't know the observation space size.
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.deep_q = None
        self.mode = mode
        self.lr = lr
        self.training_param = training_param
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
        # return np.concatenate(self.process_buffer)
        # TODO fix cases where NUM_FRAMES is not 1 !!!!


class DeepQAgent_Improve(DeepQAgent):
    def __init__(self, action_space, mode="DDQN", training_param=TrainingParam()):
        DeepQAgent.__init__(self, action_space, mode=mode, training_param=training_param)

    def convert_obs(self, observation):
        return observation.rho


class TrainAgent(object):
    def __init__(self, agent, reward_fun=RedispReward, env=None, training_param=TrainingParam()):
        self.agent = agent
        self.reward_fun = reward_fun
        self.env = env
        self.training_param = training_param
        
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
            for i in range(self.training_param.NUM_FRAMES):
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