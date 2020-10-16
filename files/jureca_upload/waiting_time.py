import pandas as pd
import AnalFunctions as af
import numpy as np
import matplotlib.pyplot as plt
import os

path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()
af.file_writer(path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var)

sl = "/"
T_test_list = lin_var[test_var2]
lattice_type = 'jule'
runs_tested = N_runs
traj_testvar2 = []

for T_test in T_test_list:
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])

    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(runs_tested)] for folder, bi in zip(folder_frame_frac, b_folder)]
    print(np.array(loc_list).shape)
    print("load trajectories " + str(T_test))
    trajectory_frame = [
        [pd.read_csv(loc, sep="\s+", header=0, comment="#") for loc in loc_list_runs if os.path.isfile(loc)] for
        loc_list_runs in loc_list]
    # print(trajectory_frame)
    traj_testvar2.append(trajectory_frame)
    print("/load trajectories")


waittime_list = []
test_var_list = []
test_var2_list = []
b_list = []
N_ped_list = []
T_list = []
os.system("mkdir " + path + "waittime")
t_min = 10
t_max = 35
t_start = t_min * fps
t_end = t_max * fps

for trajectory_frame,Ti in zip(traj_testvar2, lin_var[test_var2]):
    bi = 0
    for test_frames in trajectory_frame:
        x_list = []
        y_list = []
        ttt_list = []
        print("<calculating" + test_str + " = " + str(lin_var[test_var][bi]) +">")
        for test_frame in test_frames:
            #t_end = test_frame["FR"].max()
            for i in range(1,N_ped):
                test_frame_id = test_frame[test_frame["ID"] == i]
                if test_frame_id['Y'].min() < - 0.2:
                    test_frame_id = test_frame_id[test_frame_id['Y'] >= 0.]
                    for t in np.arange(t_start,t_end,int(fps/2)):
                        if t in test_frame_id['FR'].values and test_frame_id['Y'].min() < 0.2:
                            #print(test_frame_id['Y'].min())
                            test_frame_t = test_frame_id[test_frame["FR"] == t]
                            ttt = test_frame_id['FR'].max() - test_frame_t['FR'].values[0]
                            x = test_frame_t['X'].values[0]
                            y = test_frame_t['Y'].values[0]
                            x_list.append(x)
                            y_list.append(y)
                            ttt_list.append(ttt)
        print("<calculating" + test_str + " = " + str(lin_var[test_var][bi]) +">")
        #plt.plot(x_list,y_list)
        x_a = np.array(x_list)
        y_a = np.array(y_list)
        r_a = [round(np.sqrt(x**2 + y**2),2) for x,y in zip(x_a,y_a)]
        df = pd.DataFrame({'r':r_a,'ttt':ttt_list})
        df.sort_values(by = ['r'])
        r_max = df['r'].max()
        r_min = df['r'].min()
        #print(r_max)
        #print(r_min)
        ttt_mean_list = []
        #r_new = np.arange(r_min, r_max, 0.1)
        for r in r_a:
            ttt = df[df["r"] == r]['ttt'].values
            ttt_mean = ttt.mean()
            ttt_mean_list.append(ttt_mean)
        df_plot = pd.DataFrame({'r':r_a,'ttt':np.array(ttt_mean_list)/fps})
        file = path + "waittime/df_plot" + str(lin_var[test_var][bi]) + "_" + str(test_str2) + "_" + str(Ti) + ".csv"
        df_plot.to_csv(file)
        plt.plot(r_a,ttt_mean_list,marker = "o",linestyle='none')
        waittime_list.append(file)
        test_var_list.append(cross_var[bi][test_var])
        test_var2_list.append(cross_var[bi][test_var2])

        bi += 1
    """plt.xscale("log")
    plt.yscale("log")
    plt.show()"""
print(cross_var)
print(lin_var)
df_file = pd.DataFrame({"files": waittime_list,"test_str":test_var_list, "test_str2":test_var2_list})
df_file.to_csv(path + "waittime/file_list.csv")