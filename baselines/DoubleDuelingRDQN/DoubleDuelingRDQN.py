import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka

class DoubleDuelingRDQN(object):
    def __init__(self,
                 action_size,
                 observation_size,
                 learning_rate = 1e-5):
        self.action_size = action_size
        self.observation_size = observation_size
        self.h_size = (2 * self.action_size) + (2 * self.observation_size)

        self.lr = learning_rate
        
        self.model = None
        self.construct_q_network()

    def construct_q_network(self):
        # Defines input tensors and scalars
        self.trace_length = tf.Variable(1, dtype=tf.int32)
        input_mem_state = tfk.Input(dtype=tf.float32, shape=(self.h_size), name='input_mem_state')
        input_carry_state = tfk.Input(dtype=tf.float32, shape=(self.h_size), name='input_carry_state')
        input_layer = tfk.Input(dtype=tf.float32, shape=(None, self.observation_size), name='input_obs')

        # Forward pass
        lay1 = tfkl.Dense(self.observation_size)(input_layer)
                
        lay2 = tfkl.Dense(2 * self.observation_size)(lay1)
        lay2 = tfka.relu(lay2, alpha=0.01) #leaky_relu
        
        lay3 = tfkl.Dense(2 * self.h_size)(lay2)
        lay3 = tfka.relu(lay3, alpha=0.01) #leaky_relu
        
        lay4 = tfkl.Dense(self.h_size)(lay3)

        # Recurring part
        lstm_layer = tfkl.LSTM(self.h_size, return_state=True)
        lstm_input = lay4
        lstm_state = [input_mem_state, input_carry_state]
        lay5, mem_s, carry_s = lstm_layer(lstm_input, initial_state=lstm_state)
        lstm_output = lay5

        # Advantage and Value streams
        advantage = tfkl.Dense(2 * self.action_size)(lstm_output)
        advantage = tfka.relu(advantage, alpha=0.01) #leaky_relu
        advantage = tfkl.Dense(self.action_size)(advantage)

        value = tfkl.Dense(self.observation_size)(lstm_output)
        value = tfka.relu(value, alpha=0.01) #leaky_relu
        value = tfkl.Dense(1)(value)

        advantage_mean = tf.math.reduce_mean(advantage, axis=1, keepdims=True)
        advantage = tfkl.subtract([advantage, advantage_mean])
        Q = tf.math.add(value, advantage)

        self.model = tfk.Model(inputs=[input_mem_state, input_carry_state, input_layer],
                               outputs=[Q, mem_s, carry_s])
        losses = [
            'mse',
            self.no_loss,
            self.no_loss
        ]
        self.model.compile(loss=losses, optimizer=tfko.Adam(lr=self.lr))
        self.model.summary()

    def no_loss(self, y_true, y_pred):
        return 0.0
    
    def random_move(self, data, mem, carry):
        self.trace_length.assign(1)
        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]

        Q, mem, carry = self.model.predict(model_input, batch_size = 1) 
        move = np.random.randint(0, self.action_size)

        return move, mem, carry
        
    def predict_move(self, data, mem, carry):
        self.trace_length.assign(1)
        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]
        
        Q, mem, carry = self.model.predict(model_input, batch_size = 1)     
        move = np.argmax(Q)

        return move, Q, mem, carry

    def update_target_weights(self, target_model):
        this_weights = self.model.get_weights()
        target.model.set_weights(this_weights)

    def update_target(self, target_model, tau=1e-2):
        tau_inv = 1.0 - tau
        # Get parameters to update
        target_params = target_model.trainable_variables
        main_params = self.model.trainable_variables

        # Update each param
        for i, var in enumerate(target_params):
            var_persist = var.value() * tau_inv
            var_update = main_params[i].value() * tau
            # Poliak averaging
            var.assign(var_update + var_persist)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save(path)
        print("Successfully saved model at: {}".format(path))

    def load_network(self, path):
        # nothing has changed
        self.model.load_weights(path)
        print("Succesfully loaded network from: {}".format(path))

