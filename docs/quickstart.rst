Getting started
===================================

In this chapter we present how to install grid2op.

############
Installation
############

*************
Requirements
*************
This software uses python (at tested with version >= 3.6).

To install it i's also recommended to have `git`.

*************
Installation
*************

Using pip (recommended)
++++++++++++++++++++++++
Grid2op is hosted on pypi and can be installed like most python package with:

.. code-block:: bash

    pip install grid2op

It should be now installed. Don't hesitate to visit the section `Start Using grid2op`_ for more information on its
usage or the :ref:`grid2op-module` for a more in depth presentation of this package. If you
would rather start directly to interact with a powergrid you can visit the :ref:`make-env-module`.

From source (advanced user)
+++++++++++++++++++++++++++
If you want to develop new grid2op module (for example a new types of Backend, or a new kind of Chronics to
read new types of data) this section is made for you.


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
Start Using grid2op
####################
To get started into the grid2op ecosystem, we made a set of notebooks
that are available, without any installation thanks to
`Binder <https://mybinder.org/v2/gh/rte-france/Grid2Op/master>`_ . Feel free to visit the "getting_started" page for
more information and a detailed tour about the issue that grid2op tries to address.

The most basic code, for those familiar with openAI gym (a well-known framework in reinforcement learning) is:

.. code-block:: python

    import grid2op
    # create an environment
    env_name = "rte_case14_realistic"  # for example, other environments might be usable
    env = grid2op.make(env_name)

    # create an agent
    from grid2op.Agent import RandomAgent
    my_agent = RandomAgent(env.action_space)

    # proceed as you would any open ai gym loop
    nb_episode = 10
    for _ in range(nb_episde):
        # you perform in this case 10 different episodes
        obs = env.reset()
        reward = env.reward_range[0]
        done = False
        while not done:
            # here you loop on the time steps: at each step your agent receive an observation
            # takes an action
            # and the environment computes the next observation that will be used at the next step.
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)

**NB** Grid2Op environments implements the interface of any openAI gym environment, but they don't
inherit from them. You can use the Grid2Op environment as you would any Gym environment but they are
not strictly speaking gym environment. More information about that on the section :ref:`openai-gym`
