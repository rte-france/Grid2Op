Getting started
===================================

TODO

############
Installation
############

*************
Requirements
*************
This software uses python (at tested with version >= 3.5).

To install it i's also recommended to have `git`.

*************
Installation
*************
First, it is recommended (but optionnal) to make a virtual environment:

.. code-block:: bash

    pip3 install -U virtualenv

The second step is to clone the Grid2Op package (`git` is required):

.. code-block:: bash

    git clone https://github.com/rte-france/Grid2Op.git
    cd Grid2Op
    python3 -m virtualenv venv_grid2op

This should create a folder Grid2Op with the current sources.

Then the installation script of Grid2Op can be run to install the current simulator
(including the Python libraries dependencies):

.. code-block:: bash

    cd Grid2Op/
    source venv_grid2op/bin/activate
    pip install -U .


After this, this simulator is available under the name grid2op (from a python console)

.. code-block:: python

    import grid2op

####################
Getting started
####################
Some Jupyter notebook are provided as example of the use of the Grid2Op package. They are located in the
[getting_start](getting_started) directories.

These notebooks will help you in understanding how this framework is used and cover the most
interesting part of this framework:

* 0_basic_functionalities covers the basics
  of reinforcement learning (only the main concepts), how they are implemented in the
  Grid2Op framework. It also covers how to create a valid environment and how to use the
  `grid2op.main` function to assess how well an agent is performing.
* 1_Observation_Agents details how to create
  an "expert agent" that will take pre defined actions based on the observation it gets from
  the environment. This Notebook also covers the functioning of the BaseObservation class.
* 2_Action_GridManipulation demonstrates
  how to use the BaseAction class and how to manipulate the powergrid.
* 3_TrainingAnAgent shows how to get started with
  reinforcement learning in the Grid2Op framework. It will use the code provided by Abhinav Sagar
  available on `his blog <https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368>`_
  or on `this github repository <https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial>`_ . This code will
  be adapted (only minor changes, most of them to fit the shape of the data)
  and a (D)DQN will be trained on this problem.
* 4_StudyYourAgent shows how to study an Agent, for example
  the methods to reload a saved experiment, or to plot the powergrid given an observation for
  example. This is an introductory notebook. More user friendly graphical interface should
  come soon.

These notebooks are available without any installation thanks to
`mybinder <https://mybinder.org/v2/gh/rte-france/Grid2Op/master>`_