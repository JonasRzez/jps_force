import pandas as pd
import AnalFunctions as af
import numpy as np
path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps = af.var_ini()
sl = "/"
T_test_list = lin_var[test_var2]
lattice_type = 'jule'
runs_tested = N_runs
traj_testvar2 = []
for T_test in T_test_list:
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi,3) + "_" + str(i) + ".txt" for i in range(runs_tested)] for folder,bi in zip(folder_frame_frac,lin_var[1])]

    print("load trajectories")
    trajectory_frame = [[pd.read_csv(loc, sep="\s+", header=0, comment = "#") for loc in loc_list_runs] for loc_list_runs in loc_list]
    traj_testvar2.append(trajectory_frame)
    print("/load trajectories")

    traj_testvar2 = np.array(traj_testvar2)
    np.save("trajectories.npy",traj_testvar2)

