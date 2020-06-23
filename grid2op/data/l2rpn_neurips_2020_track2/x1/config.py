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
    "names_chronics_to_grid": None,
    "thermal_limits":[ 220.7,  334.2,  470.4,  422.1,  445.6,  427. ,    7.4,  546.3,
        566.6,  539.3,  344.8,  285.4,  591.5,  393.8,  334.6,  645. ,
        336.9,  282. ,  132.8,  182.7, 1185.1,  907.9,  400.4,  528.2,
        213. ,  336.9,  264.8,  430.2,  251. ,  473.6,  242.4,  460.6,
        317.4,  659.8,  206.5,  361.5,  321.5,  178.5,  261.6,  144.1,
        481.2,  296.2,  525.4,  201.1,  581.7,  561.8,  346.8,  486.8,
        176. ,  826. ,  546. ,  508.9,  451.5,  480.2,  294.5,  252.4,
        219.8,  316.9,  908.6,  359.1,  282.3,  280.5,  390.2,  756.7,
        554.8,  237.1,  474.2,  164.3,  202.5,  455. ,  449.4,  387.8,
        818.3,  410.2,  259.5,  203.1,  166.3,  259. ,  145.1,  258.3,
        196.7,  503.3,  446.2,  162.4,  639.1,  727.3,  115.4,  445.2,
        730.1,  253.6,  345.4,  138.2,  198.4,  248.7,  891.1, 1010.8,
        557.9,  746.2,  292.2,  150.9,  617.4,  445.5,  475. ,  200. ,
        556.5,  190.9,  188.4,  704.9,  387.8,  393.8,   43.4,  205.3,
        339.2,  204.8,  601.3,  345.4,  318.2,  678.6,  394.8,  302. ,
        329.9,  274.4,  307.6,  176.9,  352.3,  132.4,  174.7,  149.5,
         83. ,  579. ,  198.6,  557.2,  557.2,  103. ,  179.9,  196.9,
        244. ,  164.8,  100.3,  125.7,  549. ,  277.5,  273.3,   91.9,
        351.9,  307.5,  127.3,  157.5,   88. ,  108.9,  148.2,  586.2,
        129.3,  135.8,   85.1,  314.1,  207.8,  602.7,  206.5,  233.4,
        396.3,  516.9,  646.8,  651.6,  594.6,  594.6,  265.5,  223. ,
        325.2,  342.6,  307.5,  488.6,  448.4,  881.4,  579.3,  858.6,
        231.5,  423.3,  503.4,  365.4,  396.3,  270.9,  605.7,  863.4,
       1152.2,  858.6]
}
