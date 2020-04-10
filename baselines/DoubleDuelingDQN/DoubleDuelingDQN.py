# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka

class DoubleDuelingDQN(object):
    """Constructs the desired deep q learning network"""
    def __init__(self,
                 action_size,
                 observation_size,                 
                 num_frames = 1,
                 learning_rate = 1e-5):
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr = learning_rate
        self.num_frames = num_frames
        self.model = None
        self.construct_q_network()

    def construct_q_network(self):
        input_layer = tfk.Input(shape = (self.observation_size * self.num_frames,), name="input_obs")
        lay1 = tfkl.Dense(self.observation_size * self.num_frames, name="fc_1")(input_layer)
                
        lay2 = tfkl.Dense(512, name="fc_2")(lay1)
        lay2 = tfka.relu(lay2, alpha=0.01) #leaky_relu
        
        lay3 = tfkl.Dense(256, name="fc_3")(lay2)
        lay3 = tfka.relu(lay3, alpha=0.01) #leaky_relu
        
        lay4 = tfkl.Dense(128, name="fc_4")(lay3)
        lay4 = tfka.relu(lay4, alpha=0.01) #leaky_relu

        advantage = tfkl.Dense(64, name="fc_adv")(lay4)
        advantage = tfka.relu(advantage, alpha=0.01) #leaky_relu
        advantage = tfkl.Dense(self.action_size, name="adv")(advantage)

        value = tfkl.Dense(64, name="fc_val")(lay4)
        value = tfka.relu(value, alpha=0.01) #leaky_relu
        value = tfkl.Dense(1, name="val")(value)

        advantage_mean = tf.math.reduce_mean(advantage, axis=1, keepdims=True, name="adv_mean")
        advantage = tfkl.subtract([advantage, advantage_mean], name="adv_subtract")
        Q = tf.math.add(value, advantage, name="Qout")

        self.model = tfk.Model(inputs=[input_layer], outputs=[Q])
        self.model.compile(loss=self._clipped_mse_loss, optimizer=tfko.Adam(lr=self.lr))

    def _clipped_mse_loss(self, Qnext, Q):
        loss = tf.math.reduce_mean(tf.math.square(Qnext - Q), name="loss_mse")
        clipped_loss = tf.clip_by_value(loss, 0.0, 1000.0, name="loss_clip")
        return clipped_loss

    def random_move(self):
        opt_policy = np.random.randint(0, self.action_size)

        return opt_policy
        
    def predict_move(self, data):
        model_input = data.reshape(1, self.observation_size * self.num_frames)
        q_actions = self.model.predict(model_input, batch_size = 1)     
        opt_policy = np.argmax(q_actions)

        return opt_policy, q_actions[0]

    def update_target_weights(self, target_model):
        this_weights = self.model.get_weights()
        target_model.set_weights(this_weights)

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

