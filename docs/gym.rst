.. _openai-gym:

Compatibility with openAI gym
===================================

The gym framework in reinforcement learning is widely used. Starting from version 1.2.0 we improved the
compatibility with this framework.

Before grid2op 1.2.0 only some classes fully implemented the open AI gym interface:

- the :class:`grid2op.Environment` (with methods such as `env.reset`, `env.step` etc.)
- the :class:`grid2op.Agent` (with the `agent.act` etc.)
- the creation of pre defined environments (with `grid2op.make`)


Starting from 1.2.0 we implemented some automatic converters that are able to automatically map
grid2op representation for the action space and the observation space into open AI gym "spaces". More precisely these
are represented as gym.spaces.Dict.

If you are interested by this feature, we recommend you to proceed like this:

.. code-block:: python

   import grid2op
   from grid2op.Converters import GymActionSpace, GymObservationSpace
   from grid2op.Agent import BaseAgent

   class MyAgent(BaseAgent):
      def __init__(self, action_space, observation_space):
         BaseAgent.__init__(self, action_space)
         self.gym_obs_space = GymObservationSpace(observation_space)
         self.gym_action_space = GymActionSpace(observation_space)

      def act(self, obs, reward, done=False):
         # convert the observation to gym like one:
         gym_obs = self.gym_obs_space.to_gym(obs)

         # do whatever you want, as long as you retrieve a gym-like action
         gym_action = ...
         grid2op_action = self.gym_action_space.from_gym(gym_action)
         # NB advanced usage: if action_space is a grid2op.converter (for example coming from IdToAct)
         # then what's called  "grid2op_action" is in fact an action that can be understood by the converter.
         # to convert it back to grid2op action you need to convert it. See the documentation of GymActionSpace
         # for such purpose.
         return grid2op_action

   env = grid2op.make(...)
   my_agent = MyAgent(env.action_space, env.observation_space, ...)

   # and now do anything you like
   # for example
   done = False
   reward = env.reward_range[0]
   obs = env.reset()
   while not done:
      action = my_agent.act(obs, reward, done)
      obs, reward, done, info = env.step(action)

We also implemented some "converter" that allow the conversion of some action space into more convenient
`gym.spaces` (this is only available if gym is installed of course). Please check
:class:`grid2op.Converter.GymActionSpace` for more information and examples.