"""
The challenge "learning to run a power network" offers different _parameters to be customized, or to learn an
:class:`grid2op.Agent` that will perform better for example.

This class is an attempt to group them all inside one single structure.

For now, :class:`Parameters` have default value, but the can be read back / from json. Other serialization method will
come soon.
"""
import os
import json
import warnings
import re


class Parameters:
    """
    Main classes representing the _parameters of the game. The main paratemeters are describe bellow.

    Note that changing the values of these _parameters might not be enough. If these _parameters are not used in the
    :class:`grid2op.GameRules`, then modifying them will have no impact at all.

    Attributes
    ----------
    NO_OVERFLOW_DISCONNECTION: ``bool``
        If set to ``True`` then the :class:`grid2op.Environment` will not disconnect powerline above their thermal
        limit. Default is ``False``

    NB_TIMESTEP_POWERFLOW_ALLOWED: ``int``
        Number of timesteps for which a soft overflow is allowed, default 2. This means that a powerline will be
        disconnected (if :attr:`.NO_OVERFLOW_DISCONNECTION` is set to ``False``) after 2 time steps above its thermal
        limit. This is called a "soft overflow".

    NB_TIMESTEP_RECONNECTION: ``int``
        Number of timesteps a powerline disconnected for security motives (for example due to
        :attr:`.NB_TIMESTEP_POWERFLOW_ALLOWED` or :attr:`.HARD_OVERFLOW_THRESHOLD`) will remain disconnected.
        It's set to 10 timestep by default.

    NB_TIMESTEP_LINE_STATUS_REMODIF: ``int``
        When someone acts on a powerline by changing its status (connected / disconnected) this number indicates
        how many timesteps the :class:`grid2op.Agent.Agent` has to wait before being able to modify this status again.
        For examle, if this is 1, this means that an Agent can act on status of a powerline 1 out of 2 time step (1
        time step it acts, another one it cools down, and the next one it can act again). Having it at 0 it equivalent
        to deactivate this feature (default).

    NB_TIMESTEP_TOPOLOGY_REMODIF: ``int``
        When someone changes the topology of a substations, this number indicates how many timesteps the
        :class:`grid2op.Agent.Agent` has to wait before being able to modify the topology on this same substation. It
        has the same behaviour as :attr:`Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF`. To deactivate this feature,
        put it at 0 (default).

    HARD_OVERFLOW_THRESHOLD: ``float``
        If a the powerflow on a line is above HARD_OVERFLOW_THRESHOLD * thermal limit (and
        :attr:`.NO_OVERFLOW_DISCONNECTION` is set to ``False``) then it is automatically disconnected, regardless of
        the number of timesteps it is on overflow). This is called a "hard overflow"

    ENV_DC: ``bool``
        Whether or not making the simulations of the environment in the "direct current" approximation. This can be
        usefull for early training of agent, as this mode is much faster to compute than the corresponding
        "alternative current" powerflow. It is also less precise. The default is ``False``

    FORECAST_DC: ``bool``
        Whether to use the direct current approximation in the :func:`grid2op.Observation.forecasts` method. Default
        is ``False``. This can speed up the computation.

    MAX_SUB_CHANGED: ``int``
        Maximum number of substations that can be reconfigured between two consecutive timesteps by an
        :class:`grid2op.Agent`. Default value is 1.


    MAX_LINE_STATUS_CHANGED: ``int``
        Maximum number of powerlines statuses that can be changed between two consecutive timestetps by an
        :class:`grid2op.Agent`. Default value is 1.

    """
    def __init__(self, parameters_path=None):
        """
        Build an object representing the _parameters of the game.
        Parameters
        ----------
        parameters_path: ``str``, optional
            Path where to look for _parameters.
        """
        # if True, then it will not disconnect lines above their thermal limits
        self.NO_OVERFLOW_DISCONNECTION = False

        # number of timestep before powerline with an overflow is automatically disconnected
        self.NB_TIMESTEP_POWERFLOW_ALLOWED = 2

        # number of timestep before a line can be reconnected if it has suffer a forced disconnection
        self.NB_TIMESTEP_RECONNECTION = 10

        # number of timestep before a substation topology can be modified again
        self.NB_TIMESTEP_TOPOLOGY_REMODIF = 0
        self.NB_TIMESTEP_LINE_STATUS_REMODIF = 0

        # threshold above which a powerline is instantly disconnected by protections
        # this is expressed in relative value of the thermal limits
        # for example setting "HARD_OVERFLOW_THRESHOLD = 2" is equivalent, if a powerline has a thermal limit of
        # 243 A, to disconnect it instantly if it has a powerflow higher than 2 * 243 = 486 A
        self.HARD_OVERFLOW_THRESHOLD = 2

        # are the powerflow performed by the environment in DC mode (dc powerflow) or AC (ac powerflow)
        self.ENV_DC = False

        # same as above, but for the forecast states
        self.FORECAST_DC = False

        # maximum number of substations that can be change in one action
        self.MAX_SUB_CHANGED = 1

        # maximum number of powerline status that can be changed in one action
        self.MAX_LINE_STATUS_CHANGED = 1

        if parameters_path is not None:
            if os.path.isfile(parameters_path):
                if re.search(".*\.json$", parameters_path) is not None:
                    with open(parameters_path) as f:
                        dict_ = json.load(f)
                    self._init_from_json(dict_)
                else:
                    warn_msg = "Parameters: the file {} is not a supported file for loading " \
                               "parameters. Continuing with default _parameters."
                    warnings.warn(warn_msg.format(parameters_path))

            else:
                warn_msg = "Parameters: the file {} is not found. Continuing with default parameters."
                warnings.warn(warn_msg.format(parameters_path))

    @staticmethod
    def _isok_txt(arg):
        if isinstance(arg, type(True)):
            return arg
        if isinstance(arg, type("")):
            arg = arg.strip("\"")
        elif isinstance(arg, type(1)):
            arg = "{}".format(arg)

        res = False
        if arg == "True" or arg == "T" or arg == "true" or arg == "t" or str(arg) == "1":
            res = True
        elif arg == "False" or arg == "F" or arg == "false" or arg == "f" or str(arg) == "0":
            res = False
        else:
            msg = "It's ambiguous where an argument is True or False. " \
                  "Please only provide \"True\" or \"False\" and not {}"
            raise RuntimeError(msg.format(arg))
        return res

    def init_from_dict(self, dict_):
        """
        Initialize the object given a dictionary. All keys are optional. If a key is not present in the dictionnary,
        the default parameters is used.

        Parameters
        ----------
        dict_: ``dict``
            The dictionary representing the parameters to load.

        """
        if "NO_OVERFLOW_DISCONNECTION" in dict_:
            self.NO_OVERFLOW_DISCONNECTION = Parameters._isok_txt(dict_["NO_OVERFLOW_DISCONNECTION"])

        if "NB_TIMESTEP_POWERFLOW_ALLOWED" in dict_:
            self.NB_TIMESTEP_POWERFLOW_ALLOWED = int(dict_["NB_TIMESTEP_POWERFLOW_ALLOWED"])

        if "NB_TIMESTEP_RECONNECTION" in dict_:
            self.NB_TIMESTEP_RECONNECTION = int(dict_["NB_TIMESTEP_RECONNECTION"])

        if "HARD_OVERFLOW_THRESHOLD" in dict_:
            self.HARD_OVERFLOW_THRESHOLD = float(dict_["HARD_OVERFLOW_THRESHOLD"])

        if "ENV_DC" in dict_:
            self.ENV_DC = Parameters._isok_txt(dict_["ENV_DC"])

        if "FORECAST_DC" in dict_:
            self.FORECAST_DC = Parameters._isok_txt(dict_["FORECAST_DC"])

        if "MAX_SUB_CHANGED" in dict_:
            self.MAX_SUB_CHANGED = int(dict_["MAX_SUB_CHANGED"])

        if "MAX_LINE_STATUS_CHANGED" in dict_:
            self.MAX_LINE_STATUS_CHANGED = int(dict_["MAX_LINE_STATUS_CHANGED"])

        if "NB_TIMESTEP_TOPOLOGY_REMODIF" in dict_:
            self.NB_TIMESTEP_TOPOLOGY_REMODIF = int(dict_["NB_TIMESTEP_TOPOLOGY_REMODIF"])

        if "NB_TIMESTEP_LINE_STATUS_REMODIF" in dict_:
            self.NB_TIMESTEP_TOPOLOGY_REMODIF = int(dict_["NB_TIMESTEP_LINE_STATUS_REMODIF"])

        ignored_keys = dict_.keys() - self.__dict__.keys()
        if len(ignored_keys):
            warnings.warn("Parameters: The _parameters \"{}\" used to build the Grid2Op.Parameters "
                          "class are not recognized and will be ignored.".format(ignored_keys))

    def to_dict(self):
        """
        Serialize all the _parameters as a dictionnary; Usefull to write it in json format.

        Returns
        -------
        res: ``dict``
            A representation of these _parameters in the form of a dictionnary.

        """
        res = {}
        res["NO_OVERFLOW_DISCONNECTION"] = bool(self.NO_OVERFLOW_DISCONNECTION)
        res["NB_TIMESTEP_POWERFLOW_ALLOWED"] = int(self.NB_TIMESTEP_POWERFLOW_ALLOWED)
        res["NB_TIMESTEP_RECONNECTION"] = int(self.NB_TIMESTEP_RECONNECTION)
        res["HARD_OVERFLOW_THRESHOLD"] = int(self.HARD_OVERFLOW_THRESHOLD)
        res["ENV_DC"] = bool(self.ENV_DC)
        res["FORECAST_DC"] = bool(self.FORECAST_DC)
        res["MAX_SUB_CHANGED"] = int(self.MAX_SUB_CHANGED)
        res["MAX_LINE_STATUS_CHANGED"] = int(self.MAX_LINE_STATUS_CHANGED)
        res["NB_TIMESTEP_TOPOLOGY_REMODIF"] = int(self.NB_TIMESTEP_TOPOLOGY_REMODIF)
        res["NB_TIMESTEP_LINE_STATUS_REMODIF"] = int(self.NB_TIMESTEP_LINE_STATUS_REMODIF)
        return res

    @staticmethod
    def init_from_json(json_path):
        """
        Initializes the _parameters from a json path.

        Parameters
        ----------
        json_path: ``str``
            The complete (*ie.* path + filename) where the json file is located.

        Returns
        -------
        res: :class:`Parameters`
            The _parameters initialized

        """
        with open(json_path) as f:
            dict_ = json.load(f)
        res = Parameters()
        res.init_from_dict(dict_)
        return res
