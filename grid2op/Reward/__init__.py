__all__ = [
    "ConstantReward",
    "EconomicReward",
    "FlatReward",
    "IncreasingFlatReward",
    "L2RPNReward",
    "RedispReward",
    "BridgeReward",
    "CloseToOverflowReward",
    "DistanceReward",
    "GameplayReward",
    "CombinedReward",
    "RewardHelper",
    "BaseReward"
]

from grid2op.Reward.ConstantReward import ConstantReward
from grid2op.Reward.EconomicReward import EconomicReward
from grid2op.Reward.FlatReward import FlatReward
from grid2op.Reward.IncreasingFlatReward import IncreasingFlatReward
from grid2op.Reward.L2RPNReward import L2RPNReward
from grid2op.Reward.RedispReward import RedispReward
from grid2op.Reward.BridgeReward import BridgeReward
from grid2op.Reward.CloseToOverflowReward import CloseToOverflowReward
from grid2op.Reward.DistanceReward import DistanceReward
from grid2op.Reward.GameplayReward import GameplayReward
from grid2op.Reward.CombinedReward import CombinedReward
from grid2op.Reward.RewardHelper import RewardHelper
from grid2op.Reward.BaseReward import BaseReward

import warnings

class Reward(BaseReward):
    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)
        warnings.warn("Reward class has been renamed \"BaseReward\". "
                      "This class Action will be removed in future versions.",
                      category=PendingDeprecationWarning)
