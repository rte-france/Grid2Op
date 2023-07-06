from grid2op.Action import PlayableAction, PowerlineSetAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import RedispReward, AlertReward
from grid2op.Rules import RulesByArea
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import GeometricOpponentMultiArea, BaseActionBudget

try:
    from grid2op.l2rpn_utils import ActionIDF2023, ObservationIDF2023
except ImportError:
    from grid2op.Action import PlayableAction
    from grid2op.Observation import CompleteObservation
    import warnings
    warnings.warn("The grid2op version you are trying to use is too old for this environment. Please upgrade it to at least grid2op 1.9.1")
    ActionIDF2023 = PlayableAction
    ObservationIDF2023 = CompleteObservation

lines_attacked = [["26_31_106",
                   "21_22_93",
                   "17_18_88",
                   "4_10_162",
                   "12_14_68",
                   "29_37_117",
                   ],
                  ["62_58_180",
                   "62_63_160",
                   "48_50_136",
                   "48_53_141",
                   "41_48_131",
                   "39_41_121",
                   "43_44_125",
                   "44_45_126",
                   "34_35_110",
                   "54_58_154",
                   ],
                  ["74_117_81",
                   "93_95_43",
                   "88_91_33",
                   "91_92_37",
                   "99_105_62",
                   "102_104_61",  
                  ]]

opponent_attack_duration = 96  # 8 hours at maximum
attack_every_xxx_hour = 32  # can still change
average_attack_duration_hour = 2  # can still change

# after modifications for infeasibility
th_lim = [ 349.,   546.,  1151.,   581.,   743.,   613.,    69.,   801.,   731.,   953.,
          463. ,  291. ,  876. ,  649. ,  461. ,  916.,   281. ,  204. ,   97. ,  251.,
          1901.,  1356.,   601.,   793.,   351.,   509.,   409.,   566.,   339.,   899.,
          356. ,  673. ,  543. , 1313. ,  411. ,  551.,   633. ,  244. ,  589. ,  285.,
          646. ,  418. ,  479. ,  327. , 1043. ,  951.,   429. ,  871. ,  449. , 1056.,
          939. ,  946. ,  759. ,  716. ,  629. ,  486.,   409. ,  296. ,  893. ,  411.,
          99.  , 326.  , 506.  , 993.  , 646.  , 257. ,  493.  , 263.  , 323.  , 513.,
          629. ,  566. , 1379. ,  659. , 3566. ,  423.,   306. ,  479. ,  279. ,  376.,
          336. ,  836. ,  759. ,  151. , 1143. ,  851.,   236. ,  846. ,  397. ,  483.,
          559. ,  216. ,  219. ,  130. , 1533. , 1733.,   916. , 1071. ,  513. ,  289.,
          796. ,  773. ,  849. ,  359. ,  566. ,  273.,   252. , 1119. ,  535. ,  581.,
          83.  , 353.  , 541.  , 316.  ,1033.  , 379. ,  316.  ,1221.  , 599.  , 313.,
          371. ,  301. ,  346. ,  449. ,  571. ,  169.,   273. ,   88. ,  113. ,  549.,
          446. ,  589. ,  589. ,  279. ,  256. ,  157.,   195. ,  221. ,  119. ,  256.9,
          287.5,  326. ,  376.6,  179.5,  927.9,  223.,    90. ,  119. ,   75. ,   79.,
          317.9,  921. ,  236. ,  249. ,  118. ,  693.,   671. ,  453. ,  318.5,  427.2,
          689. ,  701. ,  372. ,  721. ,  616. ,  616.,   108.7,  340.2,  223. ,  384.,
          409. ,  309. ,  696. , 1393. , 1089. , 1751.,   341. ,  883. ,  791. ,  661.,
          689. ,  397. , 1019. , 2063. , 2056. , 1751., ]


this_rules = RulesByArea([[0, 1, 2, 3, 10, 11, 116, 13, 12, 14, 4, 5, 6, 15, 7, 8, 9,
                           23, 27, 28, 26, 30, 114, 113, 31, 112, 16, 29, 25, 24, 17,
                           18, 19, 20, 21, 22, 24, 71, 70, 72],
                          [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                           47, 48, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                           61, 62, 63, 66, 65],
                          [69, 73, 74, 117, 75, 76, 77, 78, 79, 80, 98, 97, 96, 95, 94, 
                           93, 99, 98, 105, 103, 104, 106, 107, 108, 111, 109, 110, 102,
                           100, 92, 91, 101, 100, 90, 89, 88, 87, 84, 83, 82, 81, 85, 86,
                           68, 67, 115]
                          ])


config = {
    "backend": PandaPowerBackend,
    "action_class": ActionIDF2023,
    "observation_class": ObservationIDF2023,
    "reward_class": RedispReward,
    "gamerules_class": this_rules,
    "chronics_class": Multifolder,
    "data_feeding_kwargs":{"gridvalueClass": GridStateFromFileWithForecastsWithMaintenance,
                           "h_forecast": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
                           },
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": th_lim,
    "opponent_budget_per_ts": 0.17 * 3.,
    "opponent_init_budget": 1000.,
    "opponent_attack_cooldown": 0,
    "opponent_attack_duration": 96,
    "opponent_action_class": PowerlineSetAction,
    "opponent_class": GeometricOpponentMultiArea,
    "opponent_budget_class": BaseActionBudget,
    "kwargs_opponent": {
        "lines_attacked": lines_attacked,
        "attack_every_xxx_hour": attack_every_xxx_hour,
        "average_attack_duration_hour": average_attack_duration_hour,
        "minimum_attack_duration_hour": 1,
        "pmax_pmin_ratio": 4
    },
    "other_rewards": {"alert": AlertReward}
}
