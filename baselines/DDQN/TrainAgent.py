import numpy as np
import tensorflow as tf

from grid2op.Reward import RedispReward
from ReplayBuffer import ReplayBuffer
        
EPSILON_DECAY = 10000
FINAL_EPSILON = 1/300
INITIAL_EPSILON = 0.1
DECAY_RATE = 0.9
REPLAY_BUFFER_SIZE = 64
UPDATE_FREQ = 4

class TrainAgent(object):
    def __init__(self, agent, env,
                 name="agent", 
                 reward_fun=RedispReward,
                 num_frames=1,
                 batch_size=8):
        self.agent = agent
        self.env = env
        self.observation_size = 0
        self.name = name
        self.reward_fun = reward_fun
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE * batch_size)
    
    def train(self, num_pre_training_steps, num_training_steps, env=None):
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        epsilon = INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        done = True

        self.tf_writer = tf.summary.create_file_writer("./logs/{}".format(self.name), name=self.name)
        
        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if done:
                # Create initial state
                obs = self.env.reset()
                state = self.agent.convert_obs(obs)
                self.observation_size = state.shape[0]
                # Init model from state
                self.agent.init_deep_q(state)
                done = False

            if step % 1000 == 0:
                print("Executing step {}".format(step))

            # Choose an action
            if step < num_pre_training_steps or np.random.rand(1) < epsilon:
                a = self.agent.deep_q.random_move()
            else:
                a, _ = self.agent.deep_q.predict_move(state)
            # Convert it to a valid action
            act = self.agent.convert_act(a)
            # Execute action
            new_obs, reward, done, info = self.env.step(act)
            new_state = self.agent.convert_obs(new_obs)
            self.reward = reward
            # Save to experience buffer
            self.replay_buffer.add(state, a, reward, done, new_state)

            # Perform training when we have enough experience in buffer
            if step > num_pre_training_steps and step > self.batch_size:
                # Slowly decay chance of random action
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

                # Perform training at given frequency
                if step % UPDATE_FREQ == 0:
                    # Sample from experience buffer
                    s_batch, a_batch, r_batch, d_batch, s1_batch = self.replay_buffer.sample(self.batch_size)
                    # Perform training
                    self.batch_train(s_batch, a_batch, r_batch, d_batch, s1_batch, step)
                    # Update target network towards primary network
                    self.agent.deep_q.update_target()

            if done:
                with self.tf_writer.as_default():
                    tf.summary.scalar("reward", total_reward, step)
                    tf.summary.scalar("alive", alive_steps, step)
                print("Lived with maximum time ", alive_steps)
                print("Earned a total of reward equal to ", total_reward)
                alive_steps = 0
                total_reward = 0                    
            else:
                total_reward += reward
                alive_steps += 1
            
            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                print("Saving Network")
                self.agent.deep_q.save_network(self.name + ".h5")

            # Iterate to next loop
            alive_steps += 1
            step += 1
            obs = new_obs
            state = new_state

        # Save model after all steps
        self.agent.deep_q.save_network(self.name + ".h5")

    def batch_train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, step):
        """Trains network to fit given parameters"""
        # nothing has changed except adapting the input shapes
        targets = np.zeros((self.batch_size, self.agent.num_action))
        
        for i in range(self.batch_size):
            model_input = s_batch[i].reshape(1, self.observation_size * self.num_frames)
            target_input = s2_batch[i].reshape(1, self.observation_size * self.num_frames)
            targets[i] = self.agent.deep_q.model.predict(model_input, batch_size = 1)
            fut_action = self.agent.deep_q.target_model.predict(target_input, batch_size = 1)

            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += 0.9 * np.max(fut_action)

        loss = self.agent.deep_q.model.train_on_batch(s_batch, targets)

        # Log the loss every 100 iterations
        if step % 100 == 0:
            with self.tf_writer.as_default():
                tf.summary.scalar("loss", loss, step)        
            print("We had a loss equal to ", loss)
