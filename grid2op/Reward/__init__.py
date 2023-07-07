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
    "_AlertCostScore",
    "_AlertTrustScore"
]

from grid2op.Reward.constantReward import ConstantReward
from grid2op.Reward.economicReward import EconomicReward
from grid2op.Reward.flatReward import FlatReward
from grid2op.Reward.increasingFlatReward import IncreasingFlatReward
from grid2op.Reward.l2RPNReward import L2RPNReward
from grid2op.Reward.redispReward import RedispReward
from grid2op.Reward.bridgeReward import BridgeReward
from grid2op.Reward.closeToOverflowReward import CloseToOverflowReward
from grid2op.Reward.distanceReward import DistanceReward
from grid2op.Reward.gameplayReward import GameplayReward
from grid2op.Reward.linesReconnectedReward import LinesReconnectedReward
from grid2op.Reward.linesCapacityReward import LinesCapacityReward
from grid2op.Reward.combinedReward import CombinedReward
from grid2op.Reward.combinedScaledReward import CombinedScaledReward
from grid2op.Reward.rewardHelper import RewardHelper
from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward.l2RPNSandBoxScore import L2RPNSandBoxScore
from grid2op.Reward.episodeDurationReward import EpisodeDurationReward
from grid2op.Reward.alarmReward import AlarmReward
from grid2op.Reward._alarmScore import _AlarmScore
from grid2op.Reward.n1Reward import N1Reward
from grid2op.Reward.l2rpn_wcci2022_scorefun import L2RPNWCCI2022ScoreFun
from grid2op.Reward.alertReward import AlertReward
from grid2op.Reward._newRenewableSourcesUsageScore import _NewRenewableSourcesUsageScore
from grid2op.Reward._alertCostScore import _AlertCostScore
from grid2op.Reward._alertTrustScore import _AlertTrustScore


import warnings


class Reward(BaseReward):
    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)
        warnings.warn(
            'Reward class has been renamed "BaseReward". '
            "This class Action will be removed in future versions.",
            category=PendingDeprecationWarning,
        )
