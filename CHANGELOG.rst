Change Log
=============
[0.6.0] - 2020-02-xx
--------------------
- [???] display the grid layout and the position of the element
- [???] fix the bug in the notebook of l2rpn2019 with the
- [???] refactoring env and obs_env
- [???] better explanation of the notebook 3 with action silently
- [???] notebooks for multi env
- [???] do something to help grid2viz to parse back action.
- [???] implement other "rewards" to look at
- [???] have something remembering the topology in the environment, and when an object is
  reconnected, and no buses are specified, then it connects it to last buses.
- [???] modeled batteries / pumped storage in grid2op (generator but that can be charged / discharged)
- [???] modeled dumps in grid2op (stuff that have a given energy max, and cannot produce more than the available energy)
- [???] add the anti-agent
- [???] Implement redispatching in simulate
- [???] simulate in MultiEnv

[0.5.5] - 2020-02-xx
---------------------
- [UPDATED] gitignore to really download the prod_charac.csv file
- [UPDATED] private member in action space and observation space (`_template_act` and `_empty_obs`)
  to make it clear it's not part of the public API.
- [ADDED] a easier way to set the thermal limits directly from the environment
- [ADDED] a new environment with redispatching capabilities (case14_redisp) including data
- [ADDED] a new class to compute the economical cost
- [ADDED] a method to check if an action is ambiguous
- [ADDED] a method to set more efficiently the id of the chronics used
- [ADDED] creation of a redispatching environment for the case14.
- [ADDED] env.step now propagate the error in info
- [ADDED] notebooks for redispatching
- [UPDATED] more information in the error when plotly and seaborn are not installed and trying to load the
  graph of the network.
- [UPDATED] make a test in action: it shouldn't be possible to assign something to busbar 3.
- [ADDED] now able to initialize a runner from a valid environment.

[0.5.4] - 2020-02-06
---------------------
- [ADDED] better handling of serialization of scenarios.

[0.5.3] - 2020-02-05
---------------------
- [ADDED] parrallel processing of the environment: evaluation in parrallel of the same agent in different environments.
- [ADDED] a way to shuffle the order in which different chronics are read from the hard drive (see MultiFolder.shuffle)
- [FIXED] utility script to push docker file
- [FIXED] some tests were not passed on the main file, because of a file ignore by git.
- [FIXED] improve stability of pandapower backend.
- [UPDATED] avoid copying the grid to build observation


[0.5.2] - 2020-01-27
---------------------
- [ADDED] Adding a utility to retrieve the starting kit L2RPN 2019 competition.
- [ADDED] Layout of the powergrid graph of the substations for both the
  `5bus_example` and the `CASE_14_L2RPN2019`.
- [FIXED] Runner skipped half the episode in some cases (sequential, even number of scenarios). Now fixed.
- [FIXED] Some typos on the notebook "getting_started\4-StudyYourAgent.ipynb".
- [FIXED] Error in the conversion of observation to dictionnary. Twice the same keys were used
  ('time_next_maintenance') for both `time_next_maintenance` and `duration_next_maintenance`.
- [UPDATED] The first chronics that is processed by a runner is not the "first" one on the hardrive
  (if sorted in alphabetical order)
- [UPDATED] Better layout of substation layout (in case of multiple nodes) in PlotGraph

[0.5.1] - 2020-01-24
--------------------
- [ADDED] extra tag 'all' to install all optional dependencies.
- [FIXED] issue in the documentation of Observation, voltages are given in kV and not V.
- [FIXED] a bug in the runner that prevented the right chronics to be read, and output wrong names
- [FIXED] a bug preventing import if plotting packages where not installed, that causes the documentation to crash.

[0.5.0] - 2020-01-23
--------------------
- [BREAKING] Action/Backend has been modified with the implementation of redispatching. If
  you used a custom backend, you'll have to implement the "redispatching" part.
- [BREAKING] with the introduction of redispatching, old action space and observation space,
  stored as json for example, will not be usable: action size and observation size
  have been modified.
- [ADDED] A converter class that allows to pre-process observation, and post-process action
  when given to an `Agent`. This allows for more flexibility in the `action_space` and
  `observation_space`.
- [ADDED] Adding another example notebook `getting_started/Example_5bus.ipynb`
- [ADDED] Adding another renderer for the live environment.
- [ADDED] Redispatching possibility for the environment
- [ADDED] More complete documentation of the representation of the powergrid
  (see documentation of `Space`)
- [FIXED] A bug in the conversion from pair unit to kv in pandapower backend. Adding some tests for that too.
- [UPDATED] More complete documentation of the Action class (with some examples)
- [UPDATED] More unit test for observations
- [UPDATED] Remove the TODO's already coded
- [UPDATED] GridStateFromFile can now read the starting date and the time interval of the chronics.
- [UPDATED] Documentation of Observation: adding the units
  (`issue 22 <https://github.com/rte-france/Grid2Op/issues/22>`_)
- [UPDATED] Notebook `getting_started/4_StudyYourAgent.ipynb` to use the converter now (much shorter and clearer)

[0.4.3] - 2020-01-20
--------------------
- [FIXED] Bug in L2RPN2019 settings, that had not been modified after the changes of version 0.4.2.

[0.4.2] - 2020-01-08
--------------------
- [BREAKING] previous saved Action Spaces and Observation Spaces (as dictionnary) are no more compatible
- [BREAKING] renaming of attributes describing the powergrid across classes for better consistency:

====================  =======================  =======================
Class Name            Old Attribute Name       New Attribute Name
====================  =======================  =======================
Backend               n_lines                  n_line
Backend               n_generators             n_gen
Backend               n_loads                  n_load
Backend               n_substations            n_sub
Backend               subs_elements            sub_info
Backend               name_loads               name_load
Backend               name_prods               name_gen
Backend               name_lines               name_line
Backend               name_subs                name_sub
Backend               lines_or_to_subid        line_or_to_subid
Backend               lines_ex_to_subid        line_ex_to_subid
Backend               lines_or_to_sub_pos      line_or_to_sub_pos
Backend               lines_ex_to_sub_pos      line_ex_to_sub_pos
Backend               lines_or_pos_topo_vect   line_or_pos_topo_vect
Backend               lines_ex_pos_topo_vect   lines_ex_pos_topo_vect
Action / Observation  _lines_or_to_subid       line_or_to_subid
Action / Observation  _lines_ex_to_subid       line_ex_to_subid
Action / Observation  _lines_or_to_sub_pos     line_or_to_sub_pos
Action / Observation  _lines_ex_to_sub_pos     line_ex_to_sub_pos
Action / Observation  _lines_or_pos_topo_vect  line_or_pos_topo_vect
Action / Observation  _lines_ex_pos_topo_vect  lines_ex_pos_topo_vect
GridValue             n_lines                  n_line
====================  =======================  =======================

- [FIXED] Runner cannot save properly action and observation (sizes are not computed properly)
  **now fixed and unit test added**
- [FIXED] Plot utility has a bug in extracting grid information.
  **now fixed**
- [FIXED] gym compatibility issue for environment
- [FIXED] checking key-word arguments in "make" function: if an invalid argument is provided,
  it now raises an error.
- [UPDATED] multiple random generator streams for observations
- [UPDATED] Refactoring of the Action and Observation Space. They now both inherit from "Space"
- [UPDATED] the getting_started notebooks to reflect these changes

[0.4.1] - 2019-12-17
--------------------
- [FIXED] Bug#14 : Nan in the observation space after switching one powerline [PandaPowerBackend]
- [UPDATED] plot now improved for buses in substations

[0.4.0] - 2019-12-04
--------------------
- [ADDED] Basic tools for plotting with the `PlotPlotly` module
- [ADDED] support of maintenance operation as well as hazards in the Observation (and appropriated tests)
- [ADDED] support for maintenance operation in the Environment (read from the chronics)
- [ADDED] example of chronics with hazards and maintenance
- [UPDATED] handling of the `AmbiguousAction` and `IllegalAction` exceptions (and appropriated tests)
- [UPDATED] various documentation, in particular the class Observation
- [UPDATED] information retrievable `Observation.state_of`

[0.3.6] - 2019-12-01
--------------------
- [ADDED] functionality to restrict action based on previous actions
  (impacts `Environment`, `GameRules` and `Parameters`)
- [ADDED] tests for the notebooks in `getting_started`
- [UPDATED] readme to properly show the docker capability
- [UPDATED] Readme with docker

[0.3.5] - 2019-11-28
--------------------
- [ADDED] serialization of the environment modifications
- [ADDED] the changelog file
- [ADDED] serialization of hazards and maintenance in actions (if any)
- [FIXED] error messages in `grid2op.GridValue.check_validity`
- [UPDATED] notebook `getting_started/4_StudyYourAgent.ipynb` to reflect these changes
