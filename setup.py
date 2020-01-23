from setuptools import setup

extras = {
   'with_pygame': ['pygame'],
    "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib_trio"],
    "plots": ["plotly", "searborn"],
    "test": ["nbformat", "jupyter_client", "jyquickhelper"]
}

setup(name='Grid2Op',
      version='0.5.0',
      description='An environment that allows to perform powergrid optimization.',
      long_description='Built with modularity in mind, this package allows to perform the same operations independantly of the software used to compute powerflow or method to generate grid states or forecasts.',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='ML powergrid optmization RL power-systems',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@rte-france.com',
      url="https://github.com/rte-france/Grid2Op",
      license='MPL',
      packages=['grid2op'],
      include_package_data=True,
      # package_data={"": ["./data/chronics/*", "./data/test_multi_chronics/1/*", "./data/test_multi_chronics/2/*",
      #                    "./data/test_multi_chronics/chronics/*", "./data/test_PandaPower/*",
      #                    "data/chronics"]},
      install_requires=["numpy", "pandas", "pandapower"],
      extras_require=extras,
      zip_safe=False,
      entry_points={'console_scripts': ['Grid2Op=grid2op.command_line:main']})
