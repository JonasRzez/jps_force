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

N_del = 10
col = ["FR", "X", "Y", "ID", "IntID"]
time_cluster_T = []
os.system("mkdir " + path + "plots")
os.system("mkdir " + path + "plots/graph")
os.system("mkdir " + path + "graph")
time = np.arange(1 * fps, 30 * fps, 1)

np_cluster_array = np.empty([len(T_test_list),lin_var[test_var].shape[0],N_runs,time.shape[0]])
print(np_cluster_array.shape)

for T_test, T_i in zip(T_test_list, range(len(T_test_list))):
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])
    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(runs_tested)] for folder, bi in zip(folder_frame_frac, b_folder)]
    bi = 0
    # count_list = []
    # time_cluster = []
    for loc_list_runs, var_i in zip(loc_list, range(len(loc_list))):
        print("<calculating " + test_str + " = " + str(2 * lin_var[test_var][bi]) + ">")
        p0count = 0
        count = 0
        incount = 0
        count_mean = 0
        #n_count = N_runs * time.shape[0]
        # node_time = []
        for loc, l_i in zip(loc_list_runs, range(len(loc_list_runs))):
            if os.path.isfile(loc) == False:
                print("WARNING: file " + loc + " not found.")
                continue
            df = pd.read_csv(loc, sep="\s+", header=0, comment="#", skipinitialspace=True, usecols=col)
            n_max_frame = df.groupby('ID')['FR'].max()[df['ID']].nlargest(N_del).index.values

            df = df[~df['ID'].isin(n_max_frame)]
            df = df[df['X'] * df['X'] + df['Y'] * df['Y'] < 2. ** 2]

            for t, t_i in zip(time, range(time.shape[0])):
                xpos = []
                ypos = []
                u = []
                v = []
                df_t = df[df["FR"] == t]
                id_list = df_t["ID"].values
                int_list = df_t[df_t["IntID"] > 0]["IntID"].values
                unique, counts = np.unique(int_list, return_counts=True)
                count_mean += counts.mean()
                np_cluster_array[T_i][var_i][l_i][t_i] = counts.mean()

        plt.plot(time / fps, np_cluster_array[T_i][var_i].mean(axis=0), label="b = " + str(2 * lin_var[test_var][bi]))
        bi += 1
        plt.legend(loc = 5 )
        #print("counts_mean = ", count_mean / n_count)

    np.save(path + "cluster" + af.b_data_name(T_test, 3) + ".npy", np_cluster_array)
    plt.savefig(path + "/plots/" + "degree" + af.b_data_name(T_test, 3) + ".png")
    plt.clf()