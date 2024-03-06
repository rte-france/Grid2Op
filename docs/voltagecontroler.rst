.. currentmodule:: grid2op.VoltageControler

.. _voltage-controler-module:


Voltage Controler
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
Powergrid are really complex objects. One of this complexity comes from the management of the voltages.

To make the difference between ensuring the safety (in terms of thermal limit and powerflow) and the voltages
magement, we decided to split these two concepts on different class.

Typically, the safety (make sure every powerline is bellow its thermal limit) is ensured by an :class:`grid2op.Agent`
and the voltage control is performed by a :class:`grid2op.VoltageControler`.

This module presents the class that can be modified to adapt (on the fly) the setpoint of the generators with
respect to the voltage magnitude.

Voltage magnitude plays a big part in real time operation process. Bad voltages can lead to different kind of problem
varying from:

- high losses (the higher the voltages, the lower the losses in general)
- equipment failures (typically if the voltages are too high)
- a really bad "quality of electricity" for consumers (if voltages is too low)
- partial or total blackout in case of voltage collapse (mainly if voltages are too low)

We wanted, in this package, to treat the voltages setpoint of the generators differently from the other
part of the game. This module exposes the main class to do this.

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.VoltageControler
    :members:
    :autosummary:

.. include:: final.rst