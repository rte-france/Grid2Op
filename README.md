# Grid2Op
Grid2Op is a plateform, built with modularity in mind, that allows to perform powergrid operation.
And that's what it stands for: Grid To Operate.

This framework allows to perform most kind of powergrid operations, from modifying the setpoint of generators,
to load shedding, performing maintenance operations or modifying the *topology* of a powergrid
to solve security issues.

This version of Grid2Op relies on an open source powerflow solver ([PandaPower](https://www.pandapower.org/)),
but is also compatible with other *Backend*. If you have at your disposal another powerflow solver, 
the documentation of [grid2op/Backend.py](grid2op/Backend.py).

Using the *Backend* based on PandaPower, this tools is able to perform 1000 timesteps 
(on a laptop, python3.6, Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz, SSD hardrive) :

- in 30s for the IEEE 14 buses test case (included in this distribution as 
[test_case14.json](grid2op/data/test_PandaPower/test_case14.json) )
- in 90s for the IEEE 118 buses test case (not included)


Official documentation: *coming soon*

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
    * [3.2 Getting Started / Examples](#getting-started--examples)
*   [4 Generate the documentation](#generate-the-documentation)
*   [5 Make tests](#make-the-tests)
*   [6 License information](#license-information)

# Installation

## Install without Docker
### Requirements:
*   Python >= 3.6

### Instructions

This instructions will install grid2op with its default PandaPower Backend implementation.

#### Step 1: Install Python3
On Debian-like systems (Ubuntu):
```bash
sudo apt-get install python3
```

On Fedora-like systems:
```bash
sudo dnf install python3
```

If you have any trouble with this step, please refer to [the official webpage of Python](https://www.python.org/downloads/release/python-366/).

#### (Optional, recommended) Step 1bis: Create a virtual environment
```bash
pip3 install -U virtualenv
python3 -m virtualenv venv_grid2op
```

#### Step 2: Clone Grid2Op
```bash
git clone https://github.com/rte-france/Grid2Op.git
```

This should create a folder Grid2Op with the current sources.

#### Step 3: Run the installation script of Grid2Op
Finally, run the following Python command to install the current simulator (including the Python libraries dependencies):
```
cd Grid2Op/
pip3 install -U .
```
After this, this simulator is available under the name grid2op (e.g. ```import grid2op```).

## Install with Docker
Support of Docker *coming soon*.

# Basic usage
## Without using Docker
Experiments can be conducted using the CLI (command line interface).

### Using CLI arguments
CLI can be used to run simulations:
```bash
python3 -m grid2op.main
```

This will evaluate a *DoNothing* policy (eg. simulating and *Agent* that does not perform
any action on the powergrid, on the IEEE case 14 for 3 epochs each of 287 time steps.)

For more information:
```bash
python3 -m grid2op.main --help
```

## Using Docker
*coming soon*

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
* has an RL-focused interface, compatible with [OpenAI-gym](https://gym.openai.com/) (Coming soon)
* parameters, game rules or type of actions are perfectly parametrizable
* can adapt to any kind of input data, in various format (might require the rewriting of a class)

## Getting Started / Examples
Some Jupyter notebook are provided as example of the use of the Grid2Op package. They are located in the 
[getting_start](getting_started) directories. 


# Generate the documentation
A copy of the documentation can be built: you will need Sphinx, a Documentation building tool, and a nice-looking custom [Sphinx theme similar to the one of readthedocs.io](https://sphinx-rtd-theme.readthedocs.io/en/latest/):
```bash
pip3 install sphinx sphinx_rtd_theme
```
This installs both the Sphinx package and the custom template. Then, the documentation can be built with the command:
```
make html
```
This will create a "documentation" subdirectory and the main entry point of the document will be located at [index.html](documentation/html/index.html).

# Make the tests
Some tests (unit test, non regression test etc.) are provided with this package. They are located at grid2op/tests.

The tests can be performed with the command:
```bash
cd grid2op/tests
python3 -m unittest discover
```

All tests should pass. Performing all the tests take roughly 5 minutes
(on a laptop, python3.6, Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz, SSD hardrive).

# License information

Copyright 2019 RTE France

    RTE: http://www.rte-france.com

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2 also available at https://www.mozilla.org/en-US/MPL/2.0/
