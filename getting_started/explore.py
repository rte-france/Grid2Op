import os
import grid2op
from grid2op.Agent import PowerLineSwitch
from grid2op.ChronicsHandler import GridStateFromFileWithForecasts, Multifolder
from grid2op.EpisodeData import EpisodeData
from grid2op.Reward import L2RPNReward
from grid2op.Runner import Runner

path_agent = "study_agent_getting_started_maintenances"
path_agent = "study_agent_getting_started_long"
path_agent = "nodisc_badagent"

episode = EpisodeData.fromdisk(
    "D:/Projects/RTE - Grid2Viz/20200127_data_scripts/20200127_agents_log/" + path_agent, "3_with_hazards")

# max_iter = 2
# CHRONICS_MLUTIEPISODE = os.path.join(
#     grid2op.CHRONICS_FODLER, "test_multi_chronics")


# scoring_function = L2RPNReward
# # make a runner
# runner = Runner(init_grid_path=grid2op.CASE_14_FILE,  # this should be the same grid as the one the agent is trained one
#                 path_chron=CHRONICS_MLUTIEPISODE,  # chronics can changed of course
#                 gridStateclass=Multifolder,  # the class of chronics can changed too
#                 gridStateclass_kwargs={"gridvalueClass": GridStateFromFileWithForecasts,
#                                        "max_iter": max_iter},  # so this can changed too
#                 names_chronics_to_backend=grid2op.NAMES_CHRONICS_TO_BACKEND,  # this also can changed
#                 agentClass=PowerLineSwitch,
#                 rewardClass=scoring_function
#                 )
# res = runner.run(path_save=path_agent, nb_episode=2)
# print("The results for the evaluated agent are:")
# for chron_name, cum_reward, nb_time_step, max_ts in res:
#     msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
#     msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
#     msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(
#         nb_time_step, max_ts)
#     print(msg_tmp)

# episode = EpisodeData.fromdisk(path_agent, indx=1)

# with open("getting_started/episode.pickle", "wb") as f:
#     pickle.dump(episode, f)

# with open("getting_started/episode.pickle", "rb") as f:
#     episode = pickle.load(f)

print(
    f"Time to next maintenance: {len(episode.observations[0].time_next_maintenance)}")
print(episode.observations[0].time_next_maintenance)
print("Duration next maintenance")
print(episode.observations[0].duration_next_maintenance)
print("Name of subs")
print(episode.name_sub)
for tstp, tstmp, act in zip(episode.timesteps, episode.timestamps, episode.actions):
    if act.as_dict():
        print(f"Timestep: {tstp}")
        print(f"Timestamp: {tstmp}")
        print(f"Action: {len(act.as_dict())}")
        print(act)
        print(act.as_dict())
