import numpy as np
import tensorflow as tf

from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from ExperienceBuffer import ExperienceBuffer
from RDoubleDuelingDQN import RDoubleDuelingDQN

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.001
EPSILON_DECAY = 1024*16
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = 128
UPDATE_FREQ = 16

class RDoubleDuelingDQNAgent(AgentWithConverter):
    def __init__(self,
                 env,
                 action_space,
                 name=__name__,
                 trace_length=1,
                 batch_size=1,
                 is_training=False,
                 lr=1e-5):
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)

        # Store constructor params
        self.env = env
        self.name = name
        self.trace_length = trace_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.lr = lr
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.mem_state = None
        self.carry_state = None

        # Declare training vars
        self.exp_buffer = None
        self.done = False
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None

        # Compute dimensions from intial state
        self.obs = self.env.reset()
        self.state = self.convert_obs(self.obs)
        self.observation_size = self.state.shape[0]
        self.action_size = self.action_space.size()

        # Load network graph
        self.Qmain = RDoubleDuelingDQN(self.action_size,
                                       self.observation_size,
                                       learning_rate = self.lr)
        # Setup inital state
        self._reset_state()
        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def _init_training(self):
        self.exp_buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE, self.batch_size, self.trace_length)
        self.done = True
        self.epoch_rewards = []
        self.epoch_alive = []
        self.Qtarget = RDoubleDuelingDQN(self.action_size,
                                         self.observation_size,
                                         learning_rate = self.lr)
    
    def _reset_state(self):
        # Initial state
        self.obs = self.env.reset()
        self.state = self.convert_obs(self.obs)
        self.done = False
        self.mem_state = np.zeros(self.Qmain.h_size)
        self.carry_state = np.zeros(self.Qmain.h_size)

    def _register_experience(self, episode_exp, episode):
        missing_obs = self.trace_length - len(episode_exp)

        if missing_obs > 0: # We are missing exp to make a trace
            exp = episode_exp[0] # Use inital state to fill out
            for missing in range(missing_obs):
                # Use do_nothing action at index 0
                self.exp_buffer.add(exp[0], 0, exp[2], exp[3], exp[4], episode)

        # Register the actual experience
        for exp in episode_exp:
            self.exp_buffer.add(exp[0], exp[1], exp[2], exp[3], exp[4], episode)

    ## Agent Interface
    def convert_obs(self, observation):
        return observation.to_vect()

    def convert_act(self, action):
        return super().convert_act(action)

    def reset(self):
        self._reset_state()

    def my_act(self, state, reward, done=False):
        self._save_current_frame(state)
        a, _ = self.Qmain.predict_move(np.array(self.frames))
        return a
    
    def load_network(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qtarget.update_weights(self.Qmain.model)

    def save_network(self, path):
        self.Qmain.save_network(path)

    ## Training Procedure
    def train(self, num_pre_training_steps, num_training_steps):
        # Loop vars
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        epsilon = INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        episode = 0
        episode_exp = []

        self.tf_writer = tf.summary.create_file_writer("./logs/{}".format(self.name), name=self.name)
        
        self._reset_state()
        # Training loop
        while step < num_steps:
            # New episode
            if self.done:
                self._reset_state()
                # Push current episode experience to experience buffer
                self._register_experience(episode_exp, episode)
                # Reset current episode experience
                episode += 1
                episode_exp = []

            if step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, epsilon))

            # Choose an action
            if step <= num_pre_training_steps:
                a, m, c = self.Qmain.random_move(self.state, self.mem_state, self.carry_state)
            elif np.random.rand(1) < epsilon: # E-greedy
                a, m, c = self.Qmain.random_move(self.state, self.mem_state, self.carry_state)
            else:
                a, _, m, c = self.Qmain.predict_move(self.state, self.mem_state, self.carry_state)

            # Update LSTM state
            self.mem_state = m
            self.carry_state = c

            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = self.env.step(act)
            new_state = self.convert_obs(new_obs)
            
            # Save to current episode experience
            episode_exp.append((self.state, a, reward, self.done, new_state))

            # Train when pre-training is over
            if step > num_pre_training_steps:
                # Slowly decay chance of random action
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

                # Perform training at given frequency
                if step % UPDATE_FREQ == 0 and self.exp_buffer.can_sample():
                    # Sample from experience buffer
                    batch = self.exp_buffer.sample()
                    # Perform training
                    self._batch_train(batch, step)
                    # Update target network towards primary network
                    self.Qmain.update_target(self.Qtarget.model)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                print("Survived [{}] steps".format(alive_steps))
                print("Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1
            
            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                self.Qmain.save_network(self.name + ".h5")

            # Iterate to next loop
            step += 1
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.Qmain.save_network(self.name + ".h5")

    def _batch_train(self, batch, step):
        """Trains network to fit given parameters"""
        Q = np.zeros((self.batch_size, self.action_size))
        batch_mem = np.zeros((self.batch_size, self.Qmain.h_size))
        batch_carry = np.zeros((self.batch_size, self.Qmain.h_size))
        
        input_size = self.observation_size
        m_data = np.vstack(batch[:, 0])
        m_data = m_data.reshape(self.batch_size, self.trace_length, input_size)
        t_data = np.reshape(np.vstack(batch[:, 4]), [self.batch_size, self.trace_length, input_size])
        t_data = t_data.reshape(self.batch_size, self.trace_length, input_size)
        m_input = [batch_mem, batch_carry, m_data]
        t_input = [batch_mem, batch_carry, t_data]

        # Batch predict
        self.Qmain.trace_length.assign(self.trace_length)
        Q, _, _ = self.Qmain.model.predict(t_input, batch_size = self.batch_size)
        self.Qtarget.trace_length.assign(self.trace_length)
        Q2, _, _ = self.Qtarget.model.predict(t_input, batch_size = self.batch_size)
        
        # Compute batch Double Q update to Qtarget
        for i in range(self.batch_size):
            idx = i * (self.trace_length - 1)
            doubleQ = Q2[i, np.argmax(Q[i])]
            a = batch[idx][1]
            r = batch[idx][2]
            d = batch[idx][3]
            Q[i, a] = r
            if d == False:
                Q[i, a] += DISCOUNT_FACTOR * doubleQ

        # Batch train
        batch_x = [batch_mem, batch_carry, m_data]
        batch_y = [Q, batch_mem, batch_carry]
        loss = self.Qmain.model.train_on_batch(batch_x, batch_y)
        loss = loss[0]

        # Log some useful metrics every 5 updates
        if step % (5 * UPDATE_FREQ) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)
            print("loss =", loss)
