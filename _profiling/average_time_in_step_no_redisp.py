
"""
Grid2op schematically does basically the following, during a “step”:

1) load the next productions / loads for all generators loads
2) compile the actions of the agents / loads modifications / opponent / maintenance / hazards into one “setpoint” for the backend
3) Ask the backend to be set to the “setpoint action” above (with “backend.apply_action”)
4) ask the backend to perform a simulation (with backend.next_grid_state which might trigger multiple times the backend.runpf function)
5) read back the internal state of the backend and convert it to an observation

This script computes these times for different grid size and different backend
"""

import warnings
import pdb
import grid2op
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
from grid2op.Backend import PandaPowerBackend
from tqdm import tqdm

TABULATE_AVAIL = False
try:
    from tabulate import tabulate
    TABULATE_AVAIL = True
except ImportError:
    print("The tabulate package is not installed. Some output might not work properly")
    
NB_TS = 1000
param = Parameters()
param.NO_OVERFLOW_DISCONNECTION = True
res = {}
res_ls = {}
res_ls_solver = {}
for env_nm in ["l2rpn_case14_sandbox", "l2rpn_neurips_2020_track1", "l2rpn_wcci_2022_dev"]:
    tmp_ = {}
    res_ls[env_nm] = {}
    res_ls_solver[env_nm] = {}
    for bk_cls, nm_ in zip([PandaPowerBackend, LightSimBackend],
                           ["pandapower", "lightsim"]):
        
        tmpp_ = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(env_nm,
                            test=True,
                            param=param,
                            backend=bk_cls()
                            )
        obs = env.reset()
        env._time_create_bk_act = 0.  # 1 and 2
        env._time_apply_act = 0.  # 1, 2, 3
        env._time_powerflow = 0.  # 4
        env._time_extract_obs = 0.  # 5
        env._time_step = 0.  # total
        total_ts = 0
        ls_preproc = 0.
        ls_acpf = 0.
        ls_post_proc = 0.
        ls_solver = 0.
        with tqdm(total=NB_TS) as pbar:
            for i in range(NB_TS):
                obs, reward, done, info = env.step(env.action_space())
                total_ts = i
                pbar.update()                    
                if done:
                    break
        tmpp_["1_2"] = env._time_create_bk_act / total_ts
        tmpp_["3"] = (env._time_apply_act - env._time_create_bk_act) / total_ts
        tmpp_["4"] = (env._time_powerflow) / total_ts
        tmpp_["5"] = (env._time_extract_obs) / total_ts
        tmpp_["total"] = (env._time_step) / total_ts
        tmp_[nm_] = tmpp_
        if isinstance(env.backend, LightSimBackend):
            # special case where I extract more information from LightSimBackend
            if hasattr(env.backend, "_timer_preproc"):
                ls_preproc += env.backend._timer_preproc
                ls_post_proc += env.backend._timer_postproc
                ls_solver += env.backend._timer_solver
            ls_acpf += env.backend.comp_time
            res_ls[env_nm][nm_] =  [ls_preproc / total_ts, ls_solver / total_ts, ls_post_proc / total_ts]
            res_ls_solver[env_nm][nm_] =  [ls_solver / total_ts, ls_acpf / total_ts]
    res[env_nm] = tmp_

tab = []
for env_nm, val in res.items():
    for sol_nm, steps in val.items():
        tmp_row = [env_nm,
                   sol_nm,
                   f'{1000. * steps["total"]:.1f}',
                   f'{1000. * steps["1_2"]:.2f}',
                   f'{1000. * steps["3"]:.3f}',
                   f'{1000. * steps["4"]:.3f}',
                   f'{1000. * steps["5"]:.3f}'
                   ]
        tab.append(tmp_row)
        
tab_ls = []
for env_nm, val in res_ls.items():
    for sol_nm, steps in val.items():
        _step1, _step2, _step3 = steps
        tmp_row = [env_nm,
                   f'{1000. * _step1:.3f}',
                   f'{1000. * _step2:.3f}',
                   f'{1000. * _step3:.3f}'
                   ]
        tab_ls.append(tmp_row)
tab_ls_solver = []
for env_nm, val in res_ls_solver.items():
    for sol_nm, steps in val.items():
        _step1, _step2 = steps
        tmp_row = [env_nm,
                   f'{1000. * _step1:.3f}',
                   f'{1000. * _step2:.3f}'
                   ]
        tab_ls_solver.append(tmp_row)
        
        
hds = ["env name", "solver name", "total (ms)", "1&2 (ms)", "3 (ms)", "4 (ms)", "5 (ms)"]        
if TABULATE_AVAIL:
    res_github_readme = tabulate(tab, headers=hds, tablefmt="github")
    print(res_github_readme)
else:
    print(tab)
print()
print()
print()
hds_ls = ["env name", "1 (ms)", "2 (ms)", "3 (ms)"]
if TABULATE_AVAIL:
    res_github_readme = tabulate(tab_ls, headers=hds_ls, tablefmt="github")
    print(res_github_readme)
else:
    print(tab_ls)
print()
print()
print()
hds_ls_solver = ["env name", "time solver (ms)", "time to solve (ms)"]
if TABULATE_AVAIL:
    res_github_readme = tabulate(tab_ls_solver, headers=hds_ls_solver, tablefmt="github")
    print(res_github_readme)
else:
    print(tab_ls_solver)
