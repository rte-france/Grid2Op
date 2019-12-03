Change Log
=============

[0.3.7] - 2019-12-04
--------------------
- [ADDED] Basic tools for plotting with the `PlotPlotly` module
- [UPDATED] handling of the `AmbiguousAction` and `IllegalAction` exceptions (and appropriated tests)
- [ADDED] support of maintenance operation as well as hazards in the observation
- [UPDATED] various documentation, in particular the class observations (and appropriated tests)

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