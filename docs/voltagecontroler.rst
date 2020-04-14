.. currentmodule:: grid2op.VoltageControler

Voltage Controler
===================================
Objectives
-----------
Powergrid are really complex objects. One of this complexity comes from the management of the voltages.

To make the difference between ensuring the safety (in terms of thermal limit and powerflow) and the voltages
magement, we decided to split these two concepts on different class.

Typically, the safety (make sure every powerline is bellow its thermal limit) is ensured by an :class:`grid2op.Agent`
and the voltage control is performed by a :class:`grid2op.VoltageControler`.

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.VoltageControler
    :members:
    :autosummary:

.. include:: final.rst