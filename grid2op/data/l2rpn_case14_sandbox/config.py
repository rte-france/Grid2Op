from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "thermal_limits": [
        541., 
        450.,  
        375.,  
        636.,  
        175.,  
        285.,  
        335.,  
        657.,  
        496.,
        827.,  
        442.,  
        641.,  
        840.,  
        156.,  
        664.,  
        235.,  
        119.,  
        179.,
        1986., 
        1572.
    ],
    "names_chronics_to_grid": None
}
