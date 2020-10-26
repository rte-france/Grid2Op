from grid2op.Action import TopologyAction
from grid2op.Reward import L2RPNReward
from grid2op.Rules import AlwaysLegal
from grid2op.Chronics import ChangeNothing
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAction,
    "observation_class": None,
    "reward_class": L2RPNReward,
    "gamerules_class": AlwaysLegal,
    "chronics_class": ChangeNothing,
    "grid_value_class": None,
    "volagecontroler_class": None,
    "thermal_limits": None,
    "names_chronics_to_grid": None
}
