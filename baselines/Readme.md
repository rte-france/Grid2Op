# Baselines repository

In this repository, some baselines for different environment will be put. It will mostly represent
the code of training the baselines.

We thank kindly all contributors.

# How to?

## Use a baseline
TODO

## Submit a code that could serve as a baseline
You can share with everyone the work you have been doing on these environments. To
do that, for now, the best way is to fork the repository https://github.com/BDonnot/grid2op
put the code of your baseline in the appropriate folder `baselines\ENVNAME\` and add some
"meta data", including but not limited to: 
- your name
- email adress
- reference to a paper (if applicable)
- the code to train the agent (if applicable)
- the code to assess the performance of the agent (if you used something different than a runner)
- the final performances of your agent

And if your agent required training, we recommend specifying:
- training time (days or hours)
- number of cores of CPU, number of GPU / TPU etc. (if applicable)
- number of steps of training
