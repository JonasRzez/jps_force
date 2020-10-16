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

col = ["FR","X","Y","ID"]
blist = 2 * lin_var[test_var]
v_0_list = []
from itertools import groupby
N_del_i = 10
N_del = 10
os.system("mkdir " + path + "flow/")
err_total_up = np.empty([0])
err_total_down = np.empty([0])
x_total = np.empty([0])
DT_total = np.empty([0])
mot_total = np.empty([0])

N_list_list = np.empty([0])
t_arange_list =  np.empty([0])
numbering =  np.empty([0])
moti =  np.empty([0])
count_run = 0
for T_test in T_test_list:
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])
    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(runs_tested)] for folder, bi in zip(folder_frame_frac, b_folder)]
    bi = 0
    exit_times = np.empty([blist.shape[0],N_runs])
    exit_times_np = np.empty([0])
    for loc_list_runs,w in zip(loc_list,range(blist.shape[0])):
        print("<calculating " + test_str + " = " + str(2 * lin_var[test_var][bi]) + ">")
        p0count = 0
        count = 0
        incount = 0
        #exit_time_array = np.empty([0])
        #exit_time_list = []
        for loc,i in zip(loc_list_runs,range(N_runs)):
            if os.path.isfile(loc) == False:
                print("WARNING: file " + loc + " not found.")
                continue
            if os.stat(loc).st_size == 0:
                print("WARNING: file" + loc + " is empty")
                continue
            df = pd.read_csv(loc, sep="\s+", header=0, comment="#",skipinitialspace=True, usecols=col)
            min_value = df.groupby('ID')['Y'].min()[df['ID']]
            #print(min_value.keys())
            min_value = min_value[min_value < -0.2]
            key_filtered = min_value.keys().values
            df = df[df['ID'].isin(key_filtered)]
            df = df[df['Y'] >= 0]
            #print(df)
            df_fr = df.groupby('ID')['FR'].max()
            df_fr = pd.DataFrame({'ID':df_fr.index.values,'FR':df_fr.values})
            #print("i = ", i)
            #if 2 * lin_var[test_var][bi] in bexp:
            if i < 15:
                frames = df_fr['FR'].values
                frame_max = df_fr['FR'].max()
                N_count = 0
                N_list = []
                t_range = np.arange(0,frame_max)
                for fr in t_range:
                    if fr in frames:
                        N_count += 1
                    N_list.append(N_count)
                if T_test == 0.1:
                    f = "r"
                else:
                    f = "b"
                plt.plot(t_range/fps,N_list,color = f)
                num = np.empty([t_range.shape[0]])
                num.fill(count_run)
                mo = np.full((t_range.shape[0]),T_test)
                numbering = np.append(numbering,num)
                moti = np.append(moti,mo)
                N_list_list = np.append(N_list_list,N_list)
                t_arange_list = np.append(t_arange_list,t_range/fps)
            count_run += 1
            n_largest = df_fr['FR'].nlargest(10)
            n_smallest = df_fr['FR'].nsmallest(10)
            #print(n_largest)
            df_fr = df_fr[~df_fr['FR'].isin(n_largest)]
            df_fr = df_fr[~df_fr['FR'].isin(n_smallest)]

            #df_fr = df_fr.nsmallest(10)
            #print(df_fr)
            #print(df_fr['ID'].values.shape)
            exit_time = np.sort(df_fr['FR'].values)/fps
            exit_time_diff = exit_time[1:] - exit_time[:-1]
            exit_times[w][i] = exit_time_diff.mean()
            exit_times_np = np.append(exit_times_np, exit_time_diff)
            #print(exit_time_diff)
            #plt.boxplot(exit_time_diff)
            #exit_time_array = np.append(exit_time_array,exit_time_diff.mean())
            #exit_time_list.append(exit_time_diff)
        #exit_times.append(exit_time_array)
            #exit_time_array[w][i] = exit_time_diff[0:30]
        exit_times_df = pd.DataFrame({"DT" : exit_times_np})
        exit_times_df.to_csv(path + "/flow/DThist" + str(T_test) + ".csv")
        mot_total = np.append(mot_total,T_test)
        bi += 1
        print("</calculating>")
  

    flat_list = exit_times
    cat = np.array([np.array([i for k in den]) for i, den in zip(np.arange(0, lin_var[test_var].shape[0]), flat_list)])
    # print(cat.flatten().shape)
    # print(density_reduced.flatten().shape)
    #print(cat.flatten().shape)
    #print(flat_list.flatten().shape)
    df = pd.DataFrame()
    df['category'] = cat.flatten()
    df['density'] = flat_list.flatten()

    mean = df.groupby('category')['density'].mean()
    p025 = df.groupby('category')['density'].quantile(0.025)
    p975 = df.groupby('category')['density'].quantile(0.975)

    plt.xlabel(af.title_var(test_var))
    plt.ylabel('\Delta T')

    #x = var_new  # np.array([var[test_var] for var in cross_var])
    if test_var == 1:
        x = 2 * lin_var[1]

    err = [mean - p025, p975 - mean]
    #plt.errorbar(x, mean, yerr=err, fmt='v', label= af.label_var(test_var2) + str(T_test), ms=10)
    err_total_up = np.append(err_total_up,err[0])
    err_total_down = np.append(err_total_down,err[0])

    x_total = np.append(x_total,x)
    DT_total = np.append(DT_total,mean)
df_out = pd.DataFrame({test_str:x_total,"DT":DT_total,"DTerr_up":err_total_up,"DTerr_down":err_total_down,test_str2:mot_total})
df_out.to_csv(path + "flow/" + "flow_err.csv")

#print(t_arange_list.shape,N_list_list.shape,moti.shape,numbering.shape)
df_exp_n = pd.DataFrame({"t":t_arange_list,"NT":N_list_list,"motivation":moti,"num": numbering})
df_exp_n.to_csv(path + "flow/NT.csv")

#plt.boxplot(exit_times.T)
#plt.legend()
#plt.plot(bexp,mean_exit_exp, ls = "None", marker = "o")
#plt.show()

#plt.scatter(blist, exit_time_array/fps)
    
