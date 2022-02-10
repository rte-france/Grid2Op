import pdb
import numpy as np
import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
env_name = "l2rpn_icaps_2021_small"
env = grid2op.make(env_name,
                   backend=LightSimBackend(),
                   data_feeding_kwargs={"max_iter": 10})
env.seed(0)
env.set_id(0)
obs = env.reset()
# pdb.set_trace()
# act = env.action_space({"curtail": [(el, 0.32) for el in np.where(env.gen_renewable)[0]]})
act = env.action_space({"curtail": [(el, 0.16) for el in np.where(env.gen_renewable)[0]]})
obs, reward, done, info = env.step(act)
assert done

# print(info["exception"])

# first test: continue to decrease
param = Parameters()
param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
env.change_parameters(param)
env.set_id(0)
obs = env.reset()
act = env.action_space({"curtail": [(el, 0.16) for el in np.where(env.gen_renewable)[0]]})
obs0, reward, done, info = env.step(act)
assert not done
print("doing nothing 1")
obs1, reward, done, info = env.step(env.action_space())
import pdb
pdb.set_trace()
assert not done
print("doing nothing 2")
obs2, reward, done, info = env.step(env.action_space())
assert not done
gen_part = env.gen_renewable & (obs2.gen_p > 0.)
assert np.all(obs2.curtailment_limit[gen_part] == obs2.curtailment[gen_part])


# second test: decrease rapidly (too much), then increase

# TODO : next : symmetric! !