# Contribution

We welcome contributions from everyone. 

If you want to contribute, a good starting point is to get in touch with us through:
- github (discussion, issues or pull-request)
- the project [discord](https://discord.gg/cYsYrPT)
- mail (see current corresponding author from the grid2op package on pypi here https://pypi.org/project/Grid2Op/) 

Contribution can take different forms:

- reporting bugs
- improving the documentation
- improving the code examples (notebooks or example in the doc)
- fixing some reported issues (for example at https://github.com/rte-france/Grid2Op/issues )
- adding a new functionality to grid2op (or increase its speed)
- extend grid2op 

# On which topic to contribute ?

If you want to contribute but are not sure were to start you can, for example:

- tackle an opened issue tagged as `good first issue` 
- try to solve an opened issue marked with `helps wanted`
- there are also some contribution ideas written in the `[TODO]` and `Next few releases` of the `CHANGELOG.rst`
  at the top level of the github repo.

In any case, if you are not sure about what is written here, feel free to ask in the grid2op [github discussion](https://github.com/orgs/Grid2op/discussions), 
in the [grid2op discord](https://discord.gg/cYsYrPT) or by contacting by mail the person in charge of the [pypi package](https://pypi.org/project/Grid2Op/).

# What to do concretely ?

For smaller changes (including, but not limited to the reporting of a bug or a contribution to the explanotory notebooks or the documentations)
a simple "pull request" with your modifications by detailing what you had in mind and the  goal of your changes.

In case of a major change (or if you have a doubt on what is "a small change"), please open an issue first
to discuss what you would like to change and then follow as closely as possible the guidelines below. This is to ensure
first that no other contribution is this direction is being currently made but also to make sure that it 
does not conflicts with some core ideas of the project.

# Guidelines for contributing to the codebase

For larger contributions, you can follow the given :

1. fork the repository located at <https://github.com/rte-france/Grid2Op>
2. synch your fork with the "latest developement branch of grid2op". For example, if the latest grid2op release
   on pypi is `1.10.3` you need to synch your repo with the branch named `dev_1.10.4` or `dev_1.11.0` (if
   the branch `dev_1.10.4` does not exist). It will be the highest number in the branches `dev_*` on
   grid2op official github repository.
3. implement your functionality / code your modifications, add documentation or any kind of contribution
4. make sure to add tests and documentation if applicable
5. once it is developed, synch your repo with the last development branch again (see point 2 above) and
   make sure to solve any possible conflicts
6. write a pull request and make sure to target the right branch (the "last development branch")

# When will it be merged ?

A contribution will be merged once approved by the maintainers (this is why it is recommended to first 
get in touch with the team)

Code in the contribution should pass all the current tests, have some dedicated tests for the new feature (if applicable)
and documentation (if applicable).

If you contribute to some grid2op notebooks, please make sure to "clear all outputs"
before making the pull request.

New contributions should respect the "**Code of Conduct**" of grid2op.
