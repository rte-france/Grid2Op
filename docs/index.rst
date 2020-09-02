.. Grid2Op documentation master file, created by
   sphinx-quickstart on Wed Jul 24 15:07:20 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Grid2Op's technical documentation!
**********************************************

Grid2Op is a pythonic, easy to use framework, to be able to develop, train or evaluate performances of "agent" or
"controler" that acts on a power grid in  different ways.

It is modular and can be use to train reinforcement learning agent or to assess the performance of optimal control
algorithm.

It is flexible and allows the power flow to be computed by the algorithm of your choice. It abstracts the modification
of a powergrid and use this abstraction to compute the **cascading failures** resulting from powerlines disconnection
for example.

**Features**

  - abstract the computation of the cascading failures
  - ability to have the same code running with multiple powerflows
  - parallel execution of one agent / controler on multiple independent scenarios (multiprocessing)
  - fully customisable: this software has been built to be fully customizable to serve different
    purposes and not only reinforcement learning, or the L2RPN competition.

Main module content
---------------------

.. toctree::
   :maxdepth: 2

   intro
   quickstart
   grid2op
   makeenv

Plotting capabilities
----------------------

.. toctree::
   :maxdepth: 2

   plot

Compatibility with openAI gym
-----------------------------
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
   my_agent = MyAgent(env.action_space, env.observation_space)

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

Technical Documentation
----------------------------

.. toctree::
   :maxdepth: 2

   action
   agent
   backend
   chronics
   converter
   environment
   rules
   observation
   opponent
   parameters
   reward
   runner
   space
   voltagecontroler

Main Exceptions
-----------------------
.. toctree::
   :maxdepth: 2

   exception

.. include:: final.rst
