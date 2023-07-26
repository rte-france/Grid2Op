.. currentmodule:: grid2op.Observation

.. _n_gen: ./space.html#grid2op.Space.GridObjects.n_gen
.. _n_load: ./space.html#grid2op.Space.GridObjects.n_load
.. _n_line: ./space.html#grid2op.Space.GridObjects.n_line
.. _n_sub: ./space.html#grid2op.Space.GridObjects.n_sub
.. _n_storage: ./space.html#grid2op.Space.GridObjects.n_storage
.. _dim_topo: ./space.html#grid2op.Space.GridObjects.dim_topo
.. _dim_alarms: ./space.html#grid2op.Space.GridObjects.dim_alarms
.. _dim_alerts: ./space.html#grid2op.Space.GridObjects.dim_alerts
.. _year: ./observation.html#grid2op.Observation.BaseObservation.year
.. _month: ./observation.html#grid2op.Observation.BaseObservation.month
.. _day: ./observation.html#grid2op.Observation.BaseObservation.day
.. _hour_of_day: ./observation.html#grid2op.Observation.BaseObservation.hour_of_day
.. _minute_of_hour: ./observation.html#grid2op.Observation.BaseObservation.minute_of_hour
.. _day_of_week: ./observation.html#grid2op.Observation.BaseObservation.day_of_week
.. _gen_p: ./observation.html#grid2op.Observation.BaseObservation.gen_p
.. _gen_q: ./observation.html#grid2op.Observation.BaseObservation.gen_q
.. _gen_v: ./observation.html#grid2op.Observation.BaseObservation.gen_v
.. _load_p: ./observation.html#grid2op.Observation.BaseObservation.load_p
.. _load_q: ./observation.html#grid2op.Observation.BaseObservation.load_q
.. _load_v: ./observation.html#grid2op.Observation.BaseObservation.load_v
.. _p_or: ./observation.html#grid2op.Observation.BaseObservation.p_or
.. _q_or: ./observation.html#grid2op.Observation.BaseObservation.q_or
.. _v_or: ./observation.html#grid2op.Observation.BaseObservation.v_or
.. _a_or: ./observation.html#grid2op.Observation.BaseObservation.a_or
.. _p_ex: ./observation.html#grid2op.Observation.BaseObservation.p_ex
.. _q_ex: ./observation.html#grid2op.Observation.BaseObservation.q_ex
.. _v_ex: ./observation.html#grid2op.Observation.BaseObservation.v_ex
.. _a_ex: ./observation.html#grid2op.Observation.BaseObservation.a_ex
.. _rho: ./observation.html#grid2op.Observation.BaseObservation.rho
.. _topo_vect: ./observation.html#grid2op.Observation.BaseObservation.topo_vect
.. _line_status: ./observation.html#grid2op.Observation.BaseObservation.line_status
.. _timestep_overflow: ./observation.html#grid2op.Observation.BaseObservation.timestep_overflow
.. _time_before_cooldown_line: ./observation.html#grid2op.Observation.BaseObservation.time_before_cooldown_line
.. _time_before_cooldown_sub: ./observation.html#grid2op.Observation.BaseObservation.time_before_cooldown_sub
.. _time_next_maintenance: ./observation.html#grid2op.Observation.BaseObservation.time_next_maintenance
.. _duration_next_maintenance: ./observation.html#grid2op.Observation.BaseObservation.duration_next_maintenance
.. _target_dispatch: ./observation.html#grid2op.Observation.BaseObservation.target_dispatch
.. _actual_dispatch: ./observation.html#grid2op.Observation.BaseObservation.actual_dispatch
.. _storage_charge: ./observation.html#grid2op.Observation.BaseObservation.storage_charge
.. _storage_power_target: ./observation.html#grid2op.Observation.BaseObservation.storage_power_target
.. _storage_power: ./observation.html#grid2op.Observation.BaseObservation.storage_power
.. _gen_p_before_curtail: ./observation.html#grid2op.Observation.BaseObservation.gen_p_before_curtail
.. _curtailment: ./observation.html#grid2op.Observation.BaseObservation.curtailment
.. _curtailment_limit: ./observation.html#grid2op.Observation.BaseObservation.curtailment_limit
.. _is_alarm_illegal: ./observation.html#grid2op.Observation.BaseObservation.is_alarm_illegal
.. _time_since_last_alarm: ./observation.html#grid2op.Observation.BaseObservation.time_since_last_alarm
.. _last_alarm: ./observation.html#grid2op.Observation.BaseObservation.last_alarm
.. _attention_budget: ./observation.html#grid2op.Observation.BaseObservation.attention_budget
.. _max_step: ./observation.html#grid2op.Observation.BaseObservation.max_step
.. _current_step: ./observation.html#grid2op.Observation.BaseObservation.current_step
.. _delta_time: ./observation.html#grid2op.Observation.BaseObservation.delta_time
.. _gen_margin_up: ./observation.html#grid2op.Observation.BaseObservation.gen_margin_up
.. _gen_margin_down: ./observation.html#grid2op.Observation.BaseObservation.gen_margin_down
.. _curtailment_mw: ./observation.html#grid2op.Observation.BaseObservation.curtailment_mw
.. _theta_or: ./observation.html#grid2op.Observation.BaseObservation.theta_or
.. _theta_ex: ./observation.html#grid2op.Observation.BaseObservation.theta_ex
.. _gen_theta: ./observation.html#grid2op.Observation.BaseObservation.gen_theta
.. _load_theta: ./observation.html#grid2op.Observation.BaseObservation.load_theta
.. _active_alert: ./observation.html#grid2op.Observation.BaseObservation.active_alert
.. _time_since_last_alert: ./observation.html#grid2op.Observation.BaseObservation.time_since_last_alert
.. _alert_duration: ./observation.html#grid2op.Observation.BaseObservation.alert_duration
.. _total_number_of_alert: ./observation.html#grid2op.Observation.BaseObservation.total_number_of_alert
.. _time_since_last_attack: ./observation.html#grid2op.Observation.BaseObservation.time_since_last_attack
.. _was_alert_used_after_attack: ./observation.html#grid2op.Observation.BaseObservation.was_alert_used_after_attack
.. _attack_under_alert: ./observation.html#grid2op.Observation.BaseObservation.attack_under_alert

.. _observation_module:

Observation
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------

In a "reinforcement learning" framework, an :class:`grid2op.Agent` receive two information before taking
any action on
the :class:`grid2op.Environment.Environment`. One of them is the :class:`grid2op.Reward.BaseReward` that tells it
how well the past action
performed. The second main input received from the environment is the :class:`BaseObservation`. This is gives the BaseAgent
partial, noisy, or complete information about the current state of the environment. This module implement a generic
:class:`BaseObservation`  class and an example of a complete observation in the case of the Learning
To Run a Power Network (`L2RPN <https://l2rpn.chalearn.org/>`_ ) competition.

Compared to other Reinforcement Learning problems the L2PRN competition allows another flexibility. Today, when
operating a powergrid, operators have "forecasts" at their disposal. We wanted to make them available in the
L2PRN competition too. In the  first edition of the L2PRN competition, was offered the
functionality to simulate the effect of an action on a forecasted powergrid.
This forecasted powergrid used:

  - the topology of the powergrid of the last know time step
  - all the injections of given in files.

This functionality was originally attached to the Environment and could only be used to simulate the effect of an action
on this unique time step. We wanted in this recoding to change that:

  - in an RL setting, an :class:`grid2op.Agent.BaseAgent` should not be able to look directly at the
    :class:`grid2op.Environment.Environment`.
    The only information about the Environment the BaseAgent should have is through the
    :class:`grid2op.Observation.BaseObservation` and
    the :class:`grid2op.Reward.BaseReward`. Having this principle implemented will help enforcing this principle.
  - In some wider context, it is relevant to have these forecasts available in multiple way, or modified by the
    :class:`grid2op.Agent.BaseAgent` itself (for example having forecast available for the next 2 or 3 hours, with
    the Agent able
    not only to change the topology of the powergrid with actions, but also the injections if he's able to provide
    more accurate predictions for example.

The :class:`BaseObservation` class implement the two above principles and is more flexible to other kind of forecasts,
or other methods to build a power grid based on the forecasts of injections.

Main observation attributes
---------------------------
In general, observations have the following attributes (if an attributes has name XXX [*eg* rho]  it can be accessed
with `obs.XXX` [*eg* `obs.rho`])

=============================================================================    ========= ==============
Name(s)                                                                          Type      Size (each)
=============================================================================    ========= ==============
`year`_, `month`_, `day`_, `hour_of_day`_, `minute_of_hour`_, `day_of_week`_     int       1
`gen_p`_, `gen_q`_, `gen_v`_, `gen_theta`_                                       float     `n_gen`_
`load_p`_, `load_q`_, `load_v`_ , `load_theta`_                                  float     `n_load`_
`p_or`_, `q_or`_, `v_or`_, `a_or`_, `theta_or`_                                  float     `n_line`_
`p_ex`_, `q_ex`_, `v_ex`_, `a_ex`_, `theta_ex`_                                  float     `n_line`_
`rho`_                                                                           float     `n_line`_
`topo_vect`_                                                                     int       `dim_topo`_
`line_status`_                                                                   bool      `n_line`_
`timestep_overflow`_                                                             int       `n_line`_
`time_before_cooldown_line`_                                                     int       `n_line`_
`time_before_cooldown_sub`_                                                      int       `n_sub`_
`time_next_maintenance`_                                                         int       `n_line`_
`duration_next_maintenance`_                                                     int       `n_line`_
`target_dispatch`_                                                               float     `n_gen`_
`actual_dispatch`_                                                               float     `n_gen`_
`storage_charge`_                                                                float     `n_storage`_
`storage_power_target`_                                                          float     `n_storage`_
`storage_power`_                                                                 float     `n_storage`_
`storage_theta`_                                                                 float     `n_storage`_
`gen_p_before_curtail`_                                                          float     `n_gen`_
`curtailment_mw`_, `curtailment`_, `curtailment_limit`_                          float     `n_gen`_
`gen_margin_up`_, `gen_margin_down`_                                             float     `n_gen`_
`is_alarm_illegal`_                                                              bool       1
`time_since_last_alarm`_                                                         int        1
`last_alarm`_                                                                    int        `dim_alarms`_
`attention_budget`_                                                              int        1
`max_step`_ , `current_step`_                                                    int        1
`delta_time`_                                                                    float      1
`total_number_of_alert`_ ,                                                       int        `dim_alerts`_
`was_alert_used_after_attack`_ , `attack_under_alert`_                           int        `dim_alerts`_
`time_since_last_alert`_ , `alert_duration`_ , `time_since_last_attack`_         int        `dim_alerts`_
`alert_duration`_                                                                bool       `dim_alerts`_
=============================================================================    ========= ==============

(*NB* for concision, if a coma ("*,*") is present in the "Name(s)" part of the column, it means multiple attributes
are present. If we take the example of the first row, it means that `obs.year`, `obs.month`, etc. are all valid
attributes of the observation, they are all integers and each is of size 1.)

.. _observation_module_graph:

But where is the graph ?
--------------------------

A powergrid can be represented as (at least) a graph (here: a mathematical object with nodes / vertices
are connected by edges).

Grid2op is made in a way that the observation and action do not explicitly represents such graph. This is
motivated first by performance reasons, but also because multiple "graphs" can represent equally
well a powergrid.

The first one that come to mind is the graph where the nodes / vertices are the buses and the edges are
the powerline. We will call this graph the "bus graph". It can be accessed in grid2op using the
:func:`grid2op.Observation.BaseObservation.bus_connectivity_matrix` for example. This will return a matrix
with 1 when 2 buses are connected and 0 otherwise, with the convention that a bus is always connected to
itself. You can think of the environments in grid2op as an environment that allows you to manipulate
this graph: split some bus in sub buses by changing at which busbar some elements are connected, or removing
some edges from this graph when powerlines are connected / disconnected. An important feature of this
graph is that its size changes: it can have a different number of nodes at different steps!

Some methods allow to retrieve these graphs, for example:

- :func:`grid2op.Observation.BaseObservation.connectivity_matrix`
- :func:`grid2op.Observation.BaseObservation.flow_bus_matrix`


For more information, you can consult the :ref:`gridgraph-module` page.

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Observation
    :members:
    :special-members:
    :autosummary:

.. include:: final.rst