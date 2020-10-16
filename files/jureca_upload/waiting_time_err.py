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


def dist(x1, y1, x2, y2, x3, y3): # x3,y3 is the point

    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    u[u > 1] = 1
    u[u < 0] = 0
    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    return np.sqrt(dx*dx + dy*dy)


"""for T_test in T_test_list:
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
"""
waittime_list_mean = []
waittime_list = []
test_var_list = []
test_var2_list = []
b_list = []
N_ped_list = []
T_list = []
os.system("mkdir " + path + "waittime")
t_min = 10
t_max = 20
t_start = t_min * fps
t_end = t_max * fps
N_del_i = 10
N_del = 0

col = [ "ID", "FR", "X", "Y"]
#t_array = np.arange(t_start, t_end, int(fps / 2))

for T_test in T_test_list:
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])

    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(runs_tested)] for folder, bi in zip(folder_frame_frac, b_folder)]
    bi = 0

    for loc_list_runs in loc_list:
        run_id = int(0)

        ttt_arr = np.empty(0)
        dist_arr = np.empty(0)
        run_id_arr = np.empty(0)
        print("<calculating " + test_str + " = " + str(lin_var[test_var][bi]) + ">")
        for loc in loc_list_runs:
            #print(loc)
            if os.path.isfile(loc) == False:
                print("file " + loc + " is missing")
                continue
            if os.stat(loc).st_size == 0:
                print("WARNING: file" + loc + " is empty")
                continue
            df = pd.read_csv(loc, sep="\s+", header=0, comment="#", skipinitialspace=True, usecols = col)

            min_value = df.groupby('ID')['Y'].min()[df['ID']]
            #print(min_value.keys())
            min_value = min_value
            min_value = min_value[min_value < -0.2]
            key_filtered = min_value.keys().values
            n_min_frame = df.groupby('ID')['FR'].min()[df['ID']].nsmallest(N_del_i).index.values

            n_max_frame = df.groupby('ID')['FR'].max()[df['ID']].nlargest(N_del).index.values

            df = df[df['ID'].isin(key_filtered)]
            df = df[df['Y'] > 0.]
            df = df[~df['ID'].isin(n_min_frame)]

            df = df[~df['ID'].isin(n_max_frame)]
            t_array = np.arange(t_start, df['FR'].max(), int(fps / 2))

            max_frame = df.groupby('ID')['FR'].max()[df['ID']]

            df['max_frame'] = max_frame.values
            df = df[df['FR'].isin(t_array)]

            #max_frame = max_frame[max_frame.index.isin(df['ID'])].sort_index()
            #N_ped_in = df['ID'].values.shape[0]
            x = df['X'].values
            y = df['Y'].values
            #dist_ttt = np.round(dist(-0.25, 0., 0.25, 0., x, y),2)
            dist_ttt = np.round(np.sqrt(x * x + y * y),2)
            ttt = df['max_frame'].values - df['FR'].values
            ttt_arr = np.append(ttt_arr, ttt)
            dist_arr = np.append(dist_arr, dist_ttt)
            run_id_arr_i = np.empty(dist_ttt.shape[0])
            run_id_arr_i.fill(int(run_id))
            run_id_arr = np.append(run_id_arr,run_id_arr_i)

            run_id += 1
        print("</calculating " + test_str + " = " + str(lin_var[test_var][bi]) + ">")
        #print("shapes,", ttt_arr.shape, dist_arr.shape, run_id_arr.shape)

        df_plot = pd.DataFrame({"ttt": ttt_arr / fps, "dist": dist_arr,"id":run_id_arr})
        df_mean = df_plot.groupby("dist").mean()

        file = path + "waittime/df_plot" + str(lin_var[test_var][bi]) + "_" + str(test_str2) + "_" + str(T_test) + ".csv"
        file_mean = path + "waittime/df_mean" + str(lin_var[test_var][bi]) + "_" + str(test_str2) + "_" + str(T_test) + ".csv"

        df_plot.to_csv(file)
        df_mean.to_csv(file_mean)
        #plt.plot(dist_ttt, ttt, marker = "o", linestyle='none')
        waittime_list.append(file)
        waittime_list_mean.append(file_mean)
        test_var_list.append(lin_var[test_var][bi])
        test_var2_list.append(T_test)

        bi += 1
    """plt.xscale("log")
    plt.yscale("log")
    plt.show()"""
print(cross_var)
print(lin_var)
df_file = pd.DataFrame({"files_mean" : waittime_list_mean, "files": waittime_list,"test_str":test_var_list, "test_str2":test_var2_list})
df_file.to_csv(path + "waittime/file_list.csv")
