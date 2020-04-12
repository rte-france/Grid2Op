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
        self.h_size = 512

        self.lr = learning_rate
        
        self.model = None
        self.construct_q_network()

    def construct_q_network(self):
        # Defines input tensors and scalars
        self.trace_length = tf.Variable(1, dtype=tf.int32, name="trace_length")
        self.dropout_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="dropout_rate")
        input_mem_state = tfk.Input(dtype=tf.float32, shape=(self.h_size), name='input_mem_state')
        input_carry_state = tfk.Input(dtype=tf.float32, shape=(self.h_size), name='input_carry_state')
        input_layer = tfk.Input(dtype=tf.float32, shape=(None, self.observation_size), name='input_obs')

        # Forward pass
        lay1 = tfkl.Dense(512, name="fc_1")(input_layer)
        # Bayesian NN simulate
        lay1 = tfkl.Dropout(self.dropout_rate, name="bnn_dropout")(lay1)

        lay2 = tfkl.Dense(256, name="fc_2")(lay1)
        lay2 = tfka.relu(lay2, alpha=0.01) #leaky_relu
        
        lay3 = tfkl.Dense(128, name="fc_3")(lay2)
        lay3 = tfka.relu(lay3, alpha=0.01) #leaky_relu
        
        lay4 = tfkl.Dense(self.h_size, name="fc_4")(lay3)
        
        # Recurring part
        lstm_layer = tfkl.LSTM(self.h_size, return_state=True, name="lstm")
        lstm_input = lay4
        lstm_state = [input_mem_state, input_carry_state]
        lay5, mem_s, carry_s = lstm_layer(lstm_input, initial_state=lstm_state)
        lstm_output = lay5
        
        # Advantage and Value streams
        advantage = tfkl.Dense(64, name="fc_adv")(lstm_output)
        advantage = tfka.relu(advantage, alpha=0.01) #leaky_relu
        advantage = tfkl.Dense(self.action_size, name="adv")(advantage)

        value = tfkl.Dense(64, name="fc_val")(lstm_output)
        value = tfka.relu(value, alpha=0.01) #leaky_relu
        value = tfkl.Dense(1, name="val")(value)

        advantage_mean = tf.math.reduce_mean(advantage, axis=1, keepdims=True, name="advantage_mean")
        advantage = tfkl.subtract([advantage, advantage_mean], name="advantage_subtract")
        Q = tf.math.add(value, advantage, name="Qout")

        # Backwards pass
        self.model = tfk.Model(inputs=[input_mem_state, input_carry_state, input_layer],
                               outputs=[Q, mem_s, carry_s])
        losses = [
            self._clipped_mse_loss,
            self._no_loss,
            self._no_loss
        ]
        self.model.compile(loss=losses, optimizer=tfko.Adam(lr=self.lr))

    def _no_loss(self, y_true, y_pred):
        return 0.0

    def _clipped_mse_loss(self, Qnext, Q):
        loss = tf.math.reduce_mean(tf.math.square(Qnext - Q), name="loss_mse")
        clipped_loss = tf.clip_by_value(loss, 0.0, 1000.0, name="loss_clip")
        return clipped_loss

    def bayesian_move(self, data, mem, carry, rate = 0.0):
        self.dropout_rate.assign(float(rate))
        self.trace_length.assign(1)
        
        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]
        
        Q, mem, carry = self.model.predict(model_input, batch_size = 1)
        move = np.argmax(Q)

        return move, Q, mem, carry
        
    def random_move(self, data, mem, carry):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]

        Q, mem, carry = self.model.predict(model_input, batch_size = 1) 
        move = np.random.randint(0, self.action_size)

        return move, mem, carry
        
    def predict_move(self, data, mem, carry):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]
        
        Q, mem, carry = self.model.predict(model_input, batch_size = 1)
        move = np.argmax(Q)

        return move, Q, mem, carry

    def update_target_hard(self, target_model):
        this_weights = self.model.get_weights()
        target_model.set_weights(this_weights)

    def update_target_soft(self, target_model, tau=1e-2):
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
        self.model.save_weights(path)
        print("Successfully saved model at: {}".format(path))

    def load_network(self, path):
        # nothing has changed
        self.model.load_weights(path)
        print("Succesfully loaded network from: {}".format(path))

