# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToActSingleElement


class ElementHandlerAgent(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 substation_id,
                 element_pos,
                 name=__name__,
                 num_frames=4,
                 is_training=False,
                 batch_size=32,
                 lr=1e-5):
        # Call parent constructor
        element_action_space_converter = IdToActSingleElement(substation_id, element_pos)
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=element_action_space_converter)
        self.obs_space = observation_space
        
        # Store constructor params
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
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = 0.0

        # Compute dimensions from intial spaces
        self.observation_size = self.obs_space.size_obs()
        self.action_size = self.action_space.size()

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=LR_DECAY_STEPS,
                                         learning_rate_decay_rate=LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

    def my_act(self, state, reward, done=False):
        return 0
