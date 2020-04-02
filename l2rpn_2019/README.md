# Learning to Run, 2019 edition

This folder present all the informations needed to reproduce the L2RPN competition locally. 
It has helpers to download the data, as well as pre-defined scripts to run locally an agent on
conditions that are as close as possible to the codalab environment.

It also has the getting started notebooks for this previous competition.

## Content of the folder

This folder contains:

- `main_l2rpn2019.py` that is a python script that emulates the behaviour the pypownet "main" function.
  It helps in setting up the grid2op platform with the default parmeters that mimick the
  L2RPN 2019 edition. You can find more information with: `python main_l2rpn2019 --help`
- `submission.py` is a script that should be used to define an BaseAgent. This agent will be 
  evaluated with the script `main_l2rpn2019.py`. You can modify it and see how well your agent performs.
- `l2rpn2019_utils` is a utility directory with some scripts that will help you download the training data, or
  used internally by some of the function define in this folder. Scripts located there should not
  be modified.
- `starting_kit` represents the port of the starting kit of the L2RPN 2019 competition into the 
  grid2op platform. One major difference is that it does not contain any data. Be careful, the
  agent that is assessed for in the starting kit is located at `starting_kit/agent_path/submission.py`
  
## Frequently asked questions

### Am I lost, what do i need to do?
A description of the competition as well as some example can be found in the notebook. We advise you
to have a look at all the notebooks in the [starting kit](starting_kit) repository and to read
the [starting kit readme](starting_kit/readme.md).

### How do i download the data?
The training data of the official l2rpn 2019 competitions can be accessed 
[here](https://github.com/BDonnot/Grid2Op/releases/tag/data_l2rpn_2019). For convenience, 
you can also directly download them from a utility script:
```commandline
python l2rpn2019_utils/download_training_data.py --path_save="data"
```
This script will download the data in the "data" folder under the name `data_l2rpn_2019`. So the
dataset will be available at `data/data_l2rpn_2019`.

### What is the size of the training set?
The whole training set is approximately 250 MB. It is made of 1004 independant chronics,
each representing the month of january for a fictive powergrid.

### How do i start training an BaseAgent?
Grid2op, and before it PypowNet, is a plateform fully compatible with an OpenAI gym
environment.

You can train any baseline you want that uses this framework. In this folder, you will get 
a pre-defined environment that corresponds to the one used for the 2019 competition.

You can have access to this environment with (python script located in this repository):
 ```python
from l2rpn2019_utils.create_env import make_l2rpn2109_env
env = make_l2rpn2109_env()
```
The `make_l2rpn2109_env` function takes one argument which is the path where the training data
are located. You can change if provided with other data sources.

Once the environment is created, you can start using it like you would any gym environment. We
recommend you look at the `getting_started` of the grid2op platform [here](../getting_started) for more information 
about the capababilities of the grid2op platform. You might find the notebook 
[4_StudyYourAgent.ipynb](../getting_started/4_StudyYourAgent.ipynb) particularly usefull.

### I am not an expert on power system. What can I do?
With the notebook [1_Power_Grid_101_notebook.ipynb](starting_kit/1_Power_Grid_101_notebook.ipynb) we 
describe the problem we want to solve.

You can also have a look at the powergrid applet at https://credc.mste.illinois.edu/applet/pg
as well as the material they provide 
[here](https://credc.mste.illinois.edu/sites/default/files/files/applets/pg_quickstart.pdf)
or [there](https://credc.mste.illinois.edu/sites/default/files/files/applets/pg_lessons.pdf).

With grid2op we tried to formalize one of the issue arising in real time power system management issue,
which is the control of flows on powerline, as a MDP (suitable for reinforcement learning) with discrete action.
You goal is to adapt the graph of the powergrid (your actions will act on a graph!) to make
sure no powerline are melting regardless of the productions and consumptions of power.

### My question is not listed here
Feel free to submit any ticket in the github project. We welcome any kind of feedback.