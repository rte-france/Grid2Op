Change Log
=============
[0.5.0] - 2020-01-xx
--------------------
- [UPDATED] more complete documentation of the Action class (with some examples)
- [ADDED] more complete documentation of the representation of the powergrid
  (see documentation of `Space`)
- [UPDATED] more unit test for observations
- [UPDATED]remove the TODO's already coded (no more todo then)
- [UPDATED] GridStateFromFile can now read the starting date and the time interval of the chronics.
- [BREAKING] Action/Backend has been modified with the implementation of redispatching. If
  you used a custom backend, you'll have to implement the "redispatching" part.
- [BREAKING] with the introduction of redispatching, old action space and observation space,
  stored as json for example, will not be usable: action size and observation size
  have been modified.

[0.4.2] - 2020-01-08
--------------------
- [FIXED] Runner cannot save properly action and observation (sizes are not computed properly)
  **now fixed and unit test added**
- [FIXED] Plot utility has a bug in extracting grid information.
  **now fixed**
- [FIXED] gym compatibility issue for environment
- [FIXED] checking key-word arguments in "make" function: if an invalid argument is provided,
  it now raises an error.
- [UPDATED] multiple random generator streams for observations
- [UPDATED] Refactoring of the Action and Observation Space. They now both ineherit from "Space"
- [BREAKING] previous saved Action Spaces and Observation Spaces (as dictionnary) are no more compatible
- [BREAKING] renaming of attributes describing the powergrid accross classes for better consistency:

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

- [UPDATE] the getting_started notebooks to reflect these changes

[0.4.1] - 2019-12-17
--------------------
- [FIXED] Bug#14 : Nan in the observation space after switching one powerline [PandaPowerBackend]
- [UPDATED] plot now improved for buses in substations

[0.4.0] - 2019-12-04
--------------------
- [ADDED] Basic tools for plotting with the `PlotPlotly` module
- [UPDATED] handling of the `AmbiguousAction` and `IllegalAction` exceptions (and appropriated tests)
- [ADDED] support of maintenance operation as well as hazards in the Observation (and appropriated tests)
- [ADDED] support for maintenance operation in the Environment (read from the chronics)
- [UPDATED] various documentation, in particular the class Observation
- [UPDATED] information retrievable `Observation.state_of`
- [ADDED] example of chronics with hazards and maintenance

[0.3.6] - 2019-12-01
--------------------
- [UPDATED] Readme with docker
- [ADDED] functionality to restrict action based on previous actions
  (impacts `Environment`, `GameRules` and `Parameters`)
- [ADDED] tests for the notebooks in `getting_started`
- [UPDATED] readme to properly show the docker capability

[0.3.5] - 2019-11-28
--------------------
- [ADDED] serialization of the environment modifications
- [FIXED] error messages in `grid2op.GridValue.check_validity`
- [ADDED] the changelog file
- [UPDATED] notebook getting_started/4_StudyYourAgent.ipynb to reflect these changes
- [ADDED] serialization of hazards and maintenance in actions (if any)