from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecastsWithMaintenance,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits":[  44.9,  205.2,  341.2,  205.8,  601.4,  347.1,  319.6,  302.8,
        330.3,  282.7,  311.2,  184.2,  354.3,  138.9,  174.9,  162.4,
         89.5,  205.7,  561.5,  561.5,  105.8,  183. ,  197.2,  244.9,
        164.9,  100.4,  125.7,  278.2,  274. ,   92.8,  353.4,  168.7,
        134.2,  158.8,   97.6,  109.9,  156.5,  140.7,  146.9,   91.3,
        318.2,  355.2,  600.7,  208.7,  233.7,  301.5,  516.7,  656.4,
        586. ,  586. ,  270.9,  230.4,  322.8,  351.4,  320.3,  841.8,
        723.5,  675.4, 1415.4]
}
