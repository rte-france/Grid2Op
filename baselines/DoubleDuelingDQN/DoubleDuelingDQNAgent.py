import numpy as np
import tensorflow as tf

from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from ReplayBuffer import ReplayBuffer
from DoubleDuelingDQN import DoubleDuelingDQN

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.001
EPSILON_DECAY = 1024*16
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = 128
UPDATE_FREQ = 16

class DoubleDuelingDQNAgent(AgentWithConverter):
    def __init__(self,
                 env,
                 action_space,
                 name=__name__,
                 num_frames=4,
                 is_training=False,
                 batch_size=32,
                 lr=1e-5):
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)

        # Store constructor params
        self.env = env
        self.name = name
        self.num_frames = num_frames
        self.is_training = is_training
        self.batch_size = batch_size
        self.lr = lr
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []

        # Declare training vars
        self.replay_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None

        # Setup training vars if needed
        if self.is_training:
            self._init_training()

        # Setup inital state
        self._reset_state()
        self._reset_frame_buffer()
        # Compute dimensions from intial state
        self.observation_size = self.state.shape[0]
        self.action_size = self.action_space.size()

        # Load network graph
        self.Qmain = DoubleDuelingDQN(self.action_size,
                                      self.observation_size,
                                      num_frames = self.num_frames,
                                      learning_rate = self.lr)
        if self.is_training:
            self.Qtarget = DoubleDuelingDQN(self.action_size,
                                            self.observation_size,
                                            num_frames = self.num_frames,
                                            learning_rate = self.lr)
    def _init_training(self):
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE * self.batch_size)
    
    def _reset_state(self):
        # Initial state
        self.obs = self.env.reset()
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = [self.state.copy() for i in range(self.num_frames)]
        if self.is_training:
            self.frames2 = [self.state.copy() for i in range(self.num_frames)]

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    ## Agent Interface
    def convert_obs(self, observation):
        return observation.to_vect()

    def convert_act(self, action):
        return super().convert_act(action)

    def reset(self):
        self._reset_state()
        self._reset_frame_buffer()

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
        # Make sure we can fill the experience buffer
        if num_pre_training_steps < self.batch_size * self.num_frames:
            num_pre_training_steps = self.batch_size * self.num_frames

        # Loop vars
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        epsilon = INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        self.done = True

        self.tf_writer = tf.summary.create_file_writer("./logs/{}".format(self.name), name=self.name)
        
        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                self._reset_state()
                self._reset_frame_buffer()
            if step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, epsilon))

            # Choose an action
            if step <= num_pre_training_steps:
                a = self.Qmain.random_move()
            elif len(self.frames) < self.num_frames:
                a = self.Qmain.random_move()
            elif np.random.rand(1) < epsilon:
                a = self.Qmain.random_move()
            else:
                a, _ = self.Qmain.predict_move(np.array(self.frames))

            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = self.env.step(act)
            new_state = self.convert_obs(new_obs)
            
            # Save to frame buffer
            self._save_current_frame(self.state)
            self._save_next_frame(new_state)

            # Save to experience buffer
            if len(self.frames) == self.num_frames:
                self.replay_buffer.add(np.array(self.frames),
                                       a, reward, self.done,
                                       np.array(self.frames2))

            # Perform training when we have enough experience in buffer
            if step > num_pre_training_steps:
                # Slowly decay chance of random action
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

                # Perform training at given frequency
                if step % UPDATE_FREQ == 0 and self.replay_buffer.size() >= self.batch_size:
                    # Sample from experience buffer
                    s_batch, a_batch, r_batch, d_batch, s1_batch = self.replay_buffer.sample(self.batch_size)
                    # Perform training
                    self._batch_train(s_batch, a_batch, r_batch, d_batch, s1_batch, step)
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

    def _batch_train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, step):
        """Trains network to fit given parameters"""
        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        m_input = np.reshape(s_batch, (self.batch_size, input_size))
        t_input = np.reshape(s2_batch, (self.batch_size, input_size))

        # Batch predict
        Q = self.Qmain.model.predict(t_input, batch_size = self.batch_size)
        Q2 = self.Qtarget.model.predict(t_input, batch_size = self.batch_size)

        # Compute batch Double Q update to Qtarget
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.model.train_on_batch(m_input, Q)

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
