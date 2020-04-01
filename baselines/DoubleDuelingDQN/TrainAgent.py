import numpy as np
import tensorflow as tf

from ReplayBuffer import ReplayBuffer
        
INITIAL_EPSILON = 0.25
FINAL_EPSILON = 0.01
EPSILON_DECAY = 10000
DISCOUNT_FACTOR = 0.975
REPLAY_BUFFER_SIZE = 128
UPDATE_FREQ = 16

class TrainAgent(object):
    def __init__(self, agent, env,
                 name="agent", 
                 num_frames=1,
                 batch_size=32):
        self.agent = agent
        self.env = env
        self.obs = self.env.reset()
        self.state = self.agent.convert_obs(self.obs)
        self.observation_size = self.state.shape[0]
        self.agent.init_deep_q(self.state)
        self.name = name
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE * batch_size)

        # Loop vars
        self.done = False
        self.frames = []
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []

    def _reset_state(self):
        # Initial state
        self.obs = self.env.reset()
        self.state = self.agent.convert_obs(self.obs)
        self.observation_size = self.state.shape[0]
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = [self.state.copy() for i in range(self.num_frames)]
        self.frames2 = [self.state.copy() for i in range(self.num_frames)]
        
    def _save_frames(self, state, next_state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)
    
    def train(self, num_pre_training_steps, num_training_steps, env=None):        
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
                a = self.agent.deep_q.random_move()
            elif len(self.frames) < self.num_frames:
                a = self.agent.deep_q.random_move()
            elif np.random.rand(1) < epsilon:
                a = self.agent.deep_q.random_move()
            else:
                a, _ = self.agent.deep_q.predict_move(np.array(self.frames))

            # Convert it to a valid action
            act = self.agent.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = self.env.step(act)
            new_state = self.agent.convert_obs(new_obs)
            
            # Save to frame buffer
            self._save_frames(self.state.copy(), new_state.copy())

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
                    # Update target network towards primary network
                    self.agent.deep_q.update_target()
                    # Sample from experience buffer
                    s_batch, a_batch, r_batch, d_batch, s1_batch = self.replay_buffer.sample(self.batch_size)
                    # Perform training
                    self.batch_train(s_batch, a_batch, r_batch, d_batch, s1_batch, step)

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
                self.agent.deep_q.save_network(self.name + ".h5")

            # Iterate to next loop
            step += 1
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.agent.deep_q.save_network(self.name + ".h5")

    def batch_train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, step):
        """Trains network to fit given parameters"""
        Q = np.zeros((self.batch_size, self.agent.num_action))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        m_input = np.reshape(s_batch, (self.batch_size, input_size))
        t_input = np.reshape(s2_batch, (self.batch_size, input_size))

        # Batch predict
        Q = self.agent.deep_q.model.predict(m_input, batch_size = self.batch_size)
        Q2 = self.agent.deep_q.target_model.predict(t_input, batch_size = self.batch_size)

        # Compute batch Double Q update to Qtarget
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.agent.deep_q.model.train_on_batch(m_input, Q)

        # Log some useful metrics every 5 updates
        if step % (5 * UPDATE_FREQ) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("loss", loss, step)
            print("loss =", loss)
