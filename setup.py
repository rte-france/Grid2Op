from setuptools import setup

extras = {
   'with_pygame': ['pygame']
}

setup(name='Grid2Op',
      version='0.2',
      description='An environment that allows to perform powergrid optimization.',
      long_description='Built with modularity in mind, this package allows to perform the same operations independantly of the software used to compute powerflow or method to generate grid states or forecasts.',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
      ],
      keywords='ML powergrid optmization RL',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@rte-france.com',
      url="https://github.com/rte-france/Grid2Op",
      license='MPL',
      packages=['grid2op'],
      include_package_data=False,
      package_data={"": ["./data/chronics/*", "./data/test_multi_chronics/1/*", "./data/test_multi_chronics/2/*",
                         "./data/test_multi_chronics/chronics/*", "./data/test_PandaPower/*"]},
      install_requires=["numpy", "pandas", "pandapower"],
      extras_require=extras,
      zip_safe=False,
      entry_points={'console_scripts': ['Grid2Op=grid2op.command_line:main']})
