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
    "LinesReconnectedReward",
    "LinesCapacityReward",
    "CombinedReward",
    "CombinedScaledReward",
    "RewardHelper",
    "BaseReward",
    "EpisodeDurationReward",
    "AlarmReward",
    "N1Reward",
    # TODO it would be better to have a specific package for this, but in the mean time i put it here
    "L2RPNSandBoxScore",
    "L2RPNWCCI2022ScoreFun",
    "AlertReward",
    "_AlarmScore",
    "_NewRenewableSourcesUsageScore",
    "_AssistantConfidenceScore",
    "_AssistantCostScore"
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
from grid2op.Reward.LinesReconnectedReward import LinesReconnectedReward
from grid2op.Reward.LinesCapacityReward import LinesCapacityReward
from grid2op.Reward.CombinedReward import CombinedReward
from grid2op.Reward.CombinedScaledReward import CombinedScaledReward
from grid2op.Reward.RewardHelper import RewardHelper
from grid2op.Reward.BaseReward import BaseReward
from grid2op.Reward.L2RPNSandBoxScore import L2RPNSandBoxScore
from grid2op.Reward.EpisodeDurationReward import EpisodeDurationReward
from grid2op.Reward.AlarmReward import AlarmReward
from grid2op.Reward._AlarmScore import _AlarmScore
from grid2op.Reward.n1Reward import N1Reward
from grid2op.Reward.l2rpn_wcci2022_scorefun import L2RPNWCCI2022ScoreFun
from grid2op.Reward.AlertReward import AlertReward
from grid2op.Reward._newRenewableSourcesUsageScore import _NewRenewableSourcesUsageScore
from grid2op.Reward._assistantScore import _AssistantConfidenceScore, _AssistantCostScore


import warnings


class Reward(BaseReward):
    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)
        warnings.warn(
            'Reward class has been renamed "BaseReward". '
            "This class Action will be removed in future versions.",
            category=PendingDeprecationWarning,
        )
