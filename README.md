# Grid2Op
[![Downloads](https://pepy.tech/badge/grid2op)](https://pepy.tech/project/grid2op)
[![PyPi_Version](https://img.shields.io/pypi/v/grid2op.svg)](https://pypi.org/project/Grid2Op/)
[![PyPi_Compat](https://img.shields.io/pypi/pyversions/grid2op.svg)](https://pypi.org/project/Grid2Op/)
[![LICENSE](https://img.shields.io/pypi/l/grid2op.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Documentation Status](https://readthedocs.org/projects/grid2op/badge/?version=latest)](https://grid2op.readthedocs.io/en/latest/?badge=latest)

Grid2Op is a plateform, built with modularity in mind, that allows to perform powergrid operation.
And that's what it stands for: Grid To Operate.

This framework allows to perform most kind of powergrid operations, from modifying the setpoint of generators,
to load shedding, performing maintenance operations or modifying the *topology* of a powergrid
to solve security issues.

This version of Grid2Op relies on an open source powerflow solver ([PandaPower](https://www.pandapower.org/)),
but is also compatible with other *Backend*. If you have at your disposal another powerflow solver, 
the documentation of [grid2op/Backend.py](grid2op/Backend.py) can help you integrate it into a proper "Backend"
and have Grid2Op using this powerflow instead of PandaPower.

Using the *Backend* based on PandaPower, this tools is able to perform 1000 timesteps 
[on a laptop, python3.6, Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz, SSD hardrive] :

- in 30s for the IEEE 14 buses test case (included in this distribution as 
[test_case14.json](grid2op/data/test_PandaPower/test_case14.json) )
- in 90s for the IEEE 118 buses test case (not included)


Official documentation: the official documentation is available at 
[https://grid2op.readthedocs.io/](https://grid2op.readthedocs.io/).

*   [1 Installation](#installation)
    *   [1.1 Install without Docker](#install-without-docker)
        *   [1.2.1 Requirements](#requirements)
        *   [1.2.2 Instructions](#instructions)
    *   [1.2 Install with Docker](#install-with-docker)
*   [2 Basic usage](#basic-usage)
    *   [2.1 Without using Docker](#without-using-docker)
    *   [2.2 Using Docker](#using-docker)
*   [3 Main features of Grid2Op](#main-features-of-grid2op)
    * [3.1 Core functionalities](#core-functionalities)
    * [3.2 Generate the documentation](#generate-the-documentation)
    * [3.3 Getting Started / Examples](#getting-started--examples)
*   [4 Make tests](#make-the-tests)
*   [5 License information](#license-information)

# Installation

## Install without Docker
### Requirements:
*   Python >= 3.6

### Instructions

This instructions will install grid2op with its default PandaPower Backend implementation.

#### Step 1: Install Python3
On Debian-like systems (Ubuntu):
```commandline
sudo apt-get install python3
```

On Fedora-like systems:
```commandline
sudo dnf install python3
```

If you have any trouble with this step, please refer to
[the official webpage of Python](https://www.python.org/downloads/release/python-366/).

#### (Optional, recommended) Step 1bis: Create a virtual environment
```commandline
pip3 install -U virtualenv
cd Grid2Op
python3 -m virtualenv venv_grid2op
```

#### Step 2: Clone Grid2Op
```commandline
git clone https://github.com/rte-france/Grid2Op.git
```

This should create a folder Grid2Op with the current sources.

#### Step 3: Run the installation script of Grid2Op
Finally, run the following Python command to install the current simulator (including the Python libraries dependencies):
```commandline
cd Grid2Op/
source venv_grid2op/bin/activate
pip install -U .
```
After this, this simulator is available under the name grid2op (e.g. ```import grid2op```).

## Install with Docker
A grid2op docker is available on [dockerhub](https://hub.docker.com/). It can be simply installed with
```commandline
docker pull bdonnot/grid2op:latest
```

This will pull and install the latest version of grid2op as a docker image. If you want a specific
version of grid2op (*eg* 0.3.3), and this version has been pushed to docker\* you can instead install:

```commandline
docker pull bdonnot/grid2op:0.3.3
```

# Basic usage
## Without using Docker
Experiments can be conducted using the CLI (command line interface).

### Using CLI arguments
CLI can be used to run simulations:
```commandline
python -m grid2op.main
```

This will evaluate a *DoNothing* policy (eg. simulating and *Agent* that does not perform
any action on the powergrid, on the IEEE case 14 for 3 epochs each of 287 time steps.)

For more information:
```commandline
python -m grid2op.main --help
```

## Using Docker
Then it's possible to start a container from the downloaded image (see [install-with-docker](#install-with-docker)):
```commandline
docker run -it bdonnot/grid2op:latest
```

This command will start a container form the image, execute the main script of grid2op 
(see [using-cli-arguments](#using-cli-arguments)) and exit this container.

If instead you want to start an interactive session, you can do:
```commandline
docker run -it bdonnot/grid2op:latest bash
```
This will start the "bash" script from the container, and you interact with it.


# Main features of Grid2Op

## Core functionalities
Built with modulartiy in mind, Grid2Op acts as a replacement of [pypownet](https://github.com/MarvinLer/pypownet) 
as a library used for the Learning To Run Power Network [L2RPN](https://l2rpn.chalearn.org/). 

Its main features are:
* emulates the behavior of a powergrid of any size at any format (provided that a *backend* is properly implemented)
* allows for grid modifications (active and reactive load values, generator voltages setpoints and active production)
* allows for maintenance operations and powergrid topological changes
* can adopt any powergrid modeling, especially Alternating Current (AC) and Direct Current (DC) approximation to 
  when performing the compitations
* supports changes of powerflow solvers, actions, observations to better suit any need in performing power system operations modeling
* has an RL-focused interface, compatible with [OpenAI-gym](https://gym.openai.com/): same interface for the
  Environment class.
* parameters, game rules or type of actions are perfectly parametrizable
* can adapt to any kind of input data, in various format (might require the rewriting of a class)

## Generate the documentation
A copy of the documentation can be built: you will need Sphinx, a Documentation building tool, and a nice-looking custom
 [Sphinx theme similar to the one of readthedocs.io](https://sphinx-rtd-theme.readthedocs.io/en/latest/):
```commandline
pip3 install -U grid2op[docs]
```
This installs both the Sphinx package and the custom template. Then, the documentation can be built with the command:
```
make html
```
This will create a "documentation" subdirectory and the main entry point of the document will be located at 
[index.html](documentation/html/index.html).

It is recommended to build this documentation locally, for convenience.
For example, the  "getting started" notebooks referenced some pages of the help.


## Getting Started / Examples
Some Jupyter notebook are provided as example of the use of the Grid2Op package. They are located in the 
[getting_start](getting_started) directories. 

These notebooks will help you in understanding how this framework is used and cover the most
interesting part of this framework:

* [0_basic_functionalities](getting_started/0_basic_functionalities.ipynb) covers the basics 
  of reinforcement learning (only the main concepts), how they are implemented in the
  Grid2Op framework. It also covers how to create a valid environment and how to use the 
  `grid2op.main` function to assess how well an agent is performing.
* [1_Observation_Agents](getting_started/1_Observation_Agents.ipynb) details how to create 
  an "expert agent" that will take pre defined actions based on the observation it gets from 
  the environment. This Notebook also covers the functioning of the Observation class.
* [2_Action_GridManipulation](getting_started/2_Action_GridManipulation.ipynb) demonstrates 
  how to use the Action class and how to manipulate the powergrid.
* [3_TrainingAnAgent](getting_started/3_TrainingAnAgent.ipynb) shows how to get started with 
  reinforcement learning in the Grid2Op framework. It will use the code provided by Abhinav Sagar
  available on [his blog](https://towardsdatascience.com/deep-reinforcement-learning-tutorial-with-open-ai-gym-c0de4471f368) 
  or on [his github repository](https://github.com/abhinavsagar/Reinforcement-Learning-Tutorial). This code will
  be adapted (only minor changes, most of them to fit the shape of the data) 
  and a (D)DQN will be trained on this problem.
* [4_StudyYourAgent](getting_started/4_StudyYourAgent.ipynb) shows how to study an Agent, for example
  the methods to reload a saved experiment, or to plot the powergrid given an observation for
  example. This is an introductory notebook. More user friendly graphical interface should
  come soon.

Try them out in your own browser without installing 
anything with the help of mybinder: 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rte-france/Grid2Op/master)


# Make the tests
Some tests (unit test, non regression test etc.) are provided with this package. They are located at grid2op/tests.

The tests can be performed with the command:
```commandline
cd grid2op/tests
python3 -m unittest discover
```

All tests should pass. Performing all the tests take roughly 5 minutes
(on a laptop, python3.6, Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz, SSD hardrive).

# License information

Copyright 2019 RTE France

    RTE: http://www.rte-france.com

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2 also available 
[here](https://www.mozilla.org/en-US/MPL/2.0/)
