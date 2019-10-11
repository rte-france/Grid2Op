from setuptools import setup

extras = {
   'with_pygame': ['pygame']
}

setup(name='Grid2Op',
      version='0.1',
      description='An environment that allows to perform powergrid optimization.',
      long_description='Built with modularity in mind, this package allows to perform the same operations independantly of the software used to compute powerflow or method to generate grid states or forecasts.',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.x',
      ],
      keywords='RL powergrid',
      url='',
      author='Benjamin DONNOT',
      author_email='Benjamin DONNOT',
      license='MPL',
      packages=['grid2op'],
      include_package_data=True,
      install_requires=["numpy", "pandas", "pandapower"],
      test_suite="setup.my_test_suite",
      extras_require=extras,
      zip_safe=False)
