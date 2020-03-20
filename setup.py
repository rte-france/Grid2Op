import subprocess
import sys
from setuptools import setup

extras = {
   'with_pygame': ['pygame'],
    "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib_trio"],
    "plots": ["plotly", "searborn", "pygame"],
    "test": ["nbformat", "jupyter_client", "jyquickhelper"]
}

all_targets = []
for el in extras:
    all_targets += extras[el]
extras["all"] = list(set(all_targets))

# try to install numba, not compatible on every platform
try:
    import numba
except (ImportError, ModuleNotFoundError):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numba"])
    except subprocess.CalledProcessError:
        print("Numba is not available for your platform. You could gain massive speed up if you could install it.")

setup(name='Grid2Op',
      version='0.5.8',
      description='An environment that allows to perform powergrid optimization.',
      long_description='Built with modularity in mind, this package allows to perform the same operations '
                       'independently of the software used to compute powerflow or method to generate grid '
                       'states or forecasts.',
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
      install_requires=["numpy", "pandas", "pandapower", "tqdm"],
      extras_require=extras,
      zip_safe=False,
      entry_points={'console_scripts': ['grid2op.main=grid2op.command_line:main',
                                        'grid2op.download=grid2op.command_line:download'
                                        ]})
