import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math as m
import os
from multiprocessing import Pool
import random as rd
from pathlib import Path
import sys
from scipy.integrate import simps
import itertools


def data_reader(simulation,folder_i,file_j):
  #  print("core = " ,folder_i , "run = ", file_j)
    folder = simulation + str(folder_i) + '/'

    str_j = str(file_j)
    file_name_y = str_j + "y.csv"
    data_y = pd.read_csv('../../c++/pedsim/traj_files/'+ simulation+ "/" + folder + file_name_y, sep=",", header=0)
    lattice_y = np.array(data_y)
    #print(lattice_y.shape)
    return lattice_y


def flow_calc(lattice_y,fps,time_int,n):
    flow_list = []
    time_int_measure = time_int
    t_array = np.arange(0,lattice_y.shape[0] - time_int_measure, time_int_measure)

    ly0 = lattice_y[0:-1]
    ly1 = lattice_y[1:]
    flow_list = np.array([np.array([1  for ly0_ii,ly1_ii in zip(ly0_i, ly1_i) if (ly0_ii>0 and ly1_ii < 0)]).sum() for ly0_i,ly1_i in zip(ly0,ly1)])
    flow_plot = np.array([flow_list[k:k+n].sum() for k in t_array])
    return flow_plot

def exit_times_fetch(lattice_y,sim_time_array,fps,time_int,n,stat_state):
    flow_list = []
    exit_times = []
    time_int_measure = time_int
    t_array = np.arange(0,lattice_y.shape[0] - time_int_measure, time_int_measure)
    ly0 = np.array(lattice_y[0:-1])
    ly1 = np.array(lattice_y[1:])
    flow_list = np.array([np.array([1 for ly0_ii,ly1_ii in zip(ly0_i, ly1_i) if (np.isnan(ly0_ii) - 1 and np.isnan(ly1_ii ))]).shape[0] for ly0_i,ly1_i in zip(ly0,ly1)])
    t_i = 0
    for flow in flow_list[:sim_time_array.shape[0]]:
        if (flow > 0 and sim_time_array[t_i] > stat_state):
            exit_times.append(sim_time_array[t_i])
        t_i += 1
    exit_times = np.array(exit_times)
    exit_times_diff = exit_times[1:] - exit_times[0:-1]
    return exit_times_diff

def kaplan_meier_estimator(exit_times_diff):
    exit_times_diff_sort = np.sort(exit_times_diff)
    max_time_diff = exit_times_diff_sort.max()
    n_i = exit_times_diff_sort.shape[0]
    survival_list = []
    d_i = 0
    t_old = 0
    survival = 1
    t_inc = 0.1
    time = np.arange(0,max_time_diff,t_inc)
    survival_list = []
    for t in time:
        t_i = t_old
        while t_i < t:
            d_i += 1
            t_i += t_inc
            survival *= (1 - d_i/n_i)
        survival_list.append(survival)
        t_old = t
    return np.array(survival_list), time
    

def multi_analysis(simulation,folder_i,file_j,time_array,fps,time_int,n,stat_state,anal_type = "none"):
    lattice_y = data_reader(simulation,folder_i,file_j)
    if anal_type == "flow" or anal_type =="all":
        flow_plot = flow_calc(lattice_y,fps,time_int,n)
        return flow_plot #exit_times
    if anal_type == "exit_diff" or anal_type =="all":
        exit_times = exit_times_fetch(lattice_y,time_array,fps,time_int,n,stat_state)
        return exit_times
    if anal_type == "survival" or anal_type =="all":
        survival = kaplan_meier_estimator(exit_times)
        return survival
    if anal_type == "none":
        print("anal_type needs specifcation which analysis to perform. choose 'flow' for flow, 'exit_diff' for difference in exit times, 'survival' for a survival plot of exit time differneces or 'all' for all mentioned")

def exit_times_flat(exit_times):
    exit_times_diff_flat = np.empty(0)
    for etd in np.array(exit_times).flatten():
        exit_times_diff_flat = np.append(exit_times_diff_flat,np.array(etd))
    return exit_times_diff_flat
def exit_times_plot(exit_times_diff_flat):
    x,bins, p = plt.hist(exit_times_diff_flat,bins = 500,density =True)
    plt.xscale('log')
    #plt.xscale('log')
    plt.show()


# In[56]:


def max_frame_fetch(traj):
    return  traj['FR'].max()

def lattice_data(traj):
    max_id = traj['ID'].max()
    max_frame = traj['FR'].max()
    #print("max_frame = ", max_frame)
    data_x_new = []
    data_y_new = []
    data_id_new = []
    data_frames_new = []
    data_angle_new = []
    for id_ped in np.arange(1, max_id + 1):
        x_i = np.array(traj[traj['ID'] == id_ped]['X'])
        x_f = np.array(traj[traj['ID'] == id_ped]['FR'])
        y_i = np.array(traj[traj['ID'] == id_ped]['Y'])
        angle_i = np.array(traj[traj['ID'] == id_ped]['ANGLE'])

        if x_i.shape[0] < max_frame:
            diff = max_frame - x_i.shape[0]
            x_nan = [np.nan for i in np.arange(0, diff)]
            x_i = np.append(x_i, x_nan)
            y_i = np.append(y_i, x_nan)
            angle_i = np.append(angle_i, x_nan)
            x_f = np.append(x_f, np.arange(x_f[-1] + 1, x_f[-1] + diff + 1))
            x_id = [id_ped for i in np.arange(0, x_i.shape[0])]
        else:
            x_id = np.array(traj[traj['ID'] == id_ped]['ID'])  # deletes the last frame of the person with maximal frames saved to unify the length of all frames
            x_id = x_id[0:-1]
            x_i = x_i[0:-1]
            angle_i = angle_i[0:-1]

            x_f = x_f[0:-1]
            y_i = y_i[0:-1]
        data_x_new = np.append(data_x_new, x_i)
        data_id_new = np.append(data_id_new, x_id)
        data_frames_new = np.append(data_frames_new, x_f)
        data_y_new = np.append(data_y_new, y_i)
        data_angle_new = np.append(data_angle_new, angle_i)
    #print("data_x_new", data_x_new.shape)
    trajectory_dic = {'id': data_id_new, 'frame': data_frames_new, 'x': data_x_new, 'y': data_y_new, 'angle': data_angle_new}
    traj_frame = pd.DataFrame(trajectory_dic)
    x_dic = {}
    y_dic = {}
    angle_dic = {}
    x_col_old_shape = 0
    #print("check befor the loop for id 99 ", traj_frame[traj_frame['id'] == 99]['x'])
    for i in np.arange(1, max_id+1):
        x_col = np.array(traj_frame[traj_frame['id'] == i]['x'])
        y_col = np.array(traj_frame[traj_frame['id'] == i]['y'])
        angle_col = np.array(traj_frame[traj_frame['id'] == i]['angle'])
        #if x_col_old_shape != x_col.shape[0]:
            #print(x_col.shape[0],x_col_old_shape)
            #print("id = ", i," wit shape ", x_col.shape[0])
         #   print(x_col)
        x_col_old_shape = x_col.shape[0]
        
        if x_col.shape[0] != max_frame:
            
            print("WARNING: x_col was not appended to shape max frame. x_col shape = ", x_col.shape, "max_frame = ", max_frame)
            diff = max_frame - x_col.shape[0]
            print("diff = ", diff)
            x_nan = [np.nan for i in np.arange(0, diff)]
            print(x_nan)
            x_col = np.append(x_col, x_nan)
            print("x_col_shape ",x_col.shape, "max_frame = ", max_frame)
            if x_col_old_shape == y_col.shape[0]:
                print("WARNING: y_col was not appended to shape max frame. x_col shape = ", y_col.shape, "max_frame = ", max_frame)
                y_col = np.append(y_col, x_nan)
            
        #print("before map shapes," ,x_col.shape, y_col.shape)

        x_dic[i] = x_col
        y_dic[i] = y_col
        angle_dic[i] = angle_col
    traj_x_frame = pd.DataFrame(x_dic)
    traj_y_frame = pd.DataFrame(y_dic)
    traj_angle_frame = pd.DataFrame(angle_dic)
 
    return traj_x_frame, traj_y_frame, traj_angle_frame


# In[4]:


def analyser(simulation,max_file,fps,file_number_start, max_folder,max_time, N_ped,door_width,stat_state  ,n,anal_type):
    flow_array = []

    #file_number_start = 0 #initial folder number
    time_int = int(1*fps) #intervall in which the flow is measured
    time_int_measure = time_int# intervall in which the flow is measured
    #max_folder = 16# max number of folders
    #max_file = 10 #max number of files
    #max_time = 3000 #max time of the simulation
    stat_state = 50 #time when the stationary state is reached
    threads = max_folder #how many threads does pool use
    
    folder_i = 0
    str_j = str(0) #initial file number
    folder = simulation + str(folder_i) + '/' #folder name
    file_name_y = str_j + "y.csv" # file name


    data_y = pd.read_csv(simulation, sep=",", header=0)
    lattice_y = np.array(data_y)
    t_array = np.arange(0,lattice_y.shape[0] - time_int_measure, time_int_measure)
    for sim_t_max in max_time:
        exit_times = np.empty(0)

        sim_time_array = np.arange(0,sim_t_max,1/fps)
        #print(sim_time_array.shape)
        for file_j in np.arange(0,max_file):
            print("calculation is at ", file_j ,'/', max_file-1 )
           # print("file =", file_j)
            #print("<pool>")
            pool = Pool(processes=threads)
            g_pool = np.array([pool.apply_async(multi_analysis, args=(simulation,folder_i,file_j,sim_time_array,fps,time_int,n,stat_state,anal_type)) for folder_i in np.arange(0,max_folder)])
            flow_plot = np.array([p.get() for p in g_pool])
            pool.close()
            #print("</pool>")
            #exit_times = np.append(exit_times,flow_plot)
            flow_array.append(flow_plot)
    return flow_array, t_array


# In[24]:


def densty1d(delta_x, a):
    return np.array(list(map(lambda x: 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2), delta_x)))


def normal(lattice_x, lattice_y, x_array, y_array, a):
    x_dens = np.array(
        [lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])

    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.mean()
    
def normal_matrix(lattice_x, lattice_y, x_array, y_array, a):
    x_dens = np.array(
        [lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])

    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.T

def b_data_name(b,dig):
    b = round(b,dig)
    str_b = ''
    for let in str(b):
        if (let in '.'):
            str_b += '_'
        else:
            str_b += let
    return str_b

def Product(variables):
    return list(itertools.product(*variables))


# In[6]:


def exit_times_fetch(lattice_y,sim_time_array,fps,time_int,stat_state):
    flow_list = []
    exit_times = []
    t_array = np.arange(0,lattice_y.shape[0] - time_int, time_int)
    ly0 = np.array(lattice_y[0:-1])
    ly1 = np.array(lattice_y[1:])
    flow_list = np.array([np.array([1 for ly0_ii,ly1_ii in zip(ly0_i, ly1_i) if (ly0_ii>-0.19 and ly1_ii < -0.19)]).shape[0] for ly0_i,ly1_i in zip(ly0,ly1)])
    t_i = 0
    for flow in flow_list[:sim_time_array.shape[0]]:
        if (flow > 0 and sim_time_array[t_i] > stat_state):
            exit_times.append(sim_time_array[t_i])
        t_i += 1
    exit_times = np.array(exit_times)
    exit_times_diff = exit_times[1:] - exit_times[0:-1]
    return exit_times_diff

def s(t,s1,s2,p):
    vec = s1 + t * (s2-s1) - p
    vec_leng = np.sqrt((vec * vec).sum())
    if vec_leng > 0:
        return vec/vec_leng
    else:
        return np.array([0,0])
def min_t(p,s1,s2):
    p1 = p-s1
    p2 = s2-s1
    p_mul = p1*p2
    p_skal = p_mul.sum(axis=1)
    leng = (s2-s1)**2
    leng = leng.sum()
    t_list = p_skal/leng
    u = np.array([s(min(max(t,0),1),s1,s2,pi) for t,pi in zip(t_list,p)])
    return u


def file_writer(path,folder_list,N_runs,b,cross_var,folder_frame,test_str,test_var):
    sl = "/"
    print("writing files")
    folder_anal_frame = [np.array(folder_frame[(folder_frame[test_str] == var[test_var])]['ini_folder'])[0] for var in
                         cross_var]
    print(folder_anal_frame)
    for i in range(N_runs):
        location = np.array(
            ["evac_traj_" + b_data_name(2 * b[var], 3) + "_" + str(i) + ".txt" for var in range(cross_var.shape[0])])
        for loc, run_folder in zip(location, folder_list):

            file_name = path + run_folder + sl + loc
            if os.path.isfile(file_name):
                file = open(file_name, 'r')
                line_count = 0
                new_file = open(path + run_folder + sl + "new_" + loc, 'w')
                # print(loc)
                for line in file:
                    line_count += 1
                    # if line_count > 10:
                    if line_count == 13:
                        new_file.write(line[1:])
                    else:
                        new_file.write(line)

                os.system("rm " + path + run_folder + sl + loc)
                new_file.close()
            else:
                print("WARNING: file " + file_name + " not found")
    print("/writing files")


def var_ini():
    sl = "/"
    path = pd.read_csv("path.csv")
    path = path['path'][0]
    variables = pd.read_csv(path + "variables_list.csv")
    var_keys = variables.keys()
    test_var = variables['test_var'][0]
    test_str = variables['test_str'][0]
    sec_test_var = False
    if "test_var2" in var_keys:
        test_var2 = variables['test_var2'][0]
        test_str2 = variables['test_str2'][0]
        sec_test_var = True
    else:
        test_var2 = 1
    append = False
    print("current path is ", path)
    folder_frame = pd.read_csv(path + "folder_list.csv")
    folder_key = folder_frame.keys()
    print(folder_key)
    folder_list = np.array(folder_frame['ini_folder'])

    folder = "trajectories/"

    testvar_list = pd.read_csv(path + test_str + "_list.csv")
    testvar_list = np.array(testvar_list)
    test_var_shape = testvar_list.shape[0]

    esigma_frame = frame_reader( folder_frame, "esigma")
    esigma = np.array(esigma_frame)

    T_frame = frame_reader( folder_frame, "T")
    print(T_frame)
    T = np.array(T_frame)

    if "mot_frac" in folder_key:
        mot_frac = frame_reader(folder_frame,"mot_frac").to_numpy
    else:
        mot_frac = 1
    print(T)
    v0_frame = frame_reader(folder_frame, "v0")
    v0 = np.array(v0_frame)

    rho_ini_frame = frame_reader(folder_frame, "rho")
    rho_ini = np.array(rho_ini_frame)

    b_frame = frame_reader( folder_frame, "b")

    N_runs = variables['N_runs'][0]
    fps = variables['fps'][0]
    b = np.array(b_frame)
    N_ped = variables['N_ped'][0]
    t_max = variables['t_max'][0]

    cross_var = np.load(path + "cross_var.npy")
    lin_var = np.load(path + "var.npy", allow_pickle=True)
    print(cross_var)
    print(folder_frame)
    print(rho_ini)

    print("b/2 = ", b)
    print("fps = ", fps)
    print("N_runs = ", N_runs)
    print("folder_list_0", folder_list[0])
    print("folder_list_-1", folder_list[-1])
    print("N_ped = ", N_ped)
    print("t_max = ", t_max)
    print("mot_frac = ", mot_frac)

    return path,folder_list,N_runs,b,cross_var,folder_frame,test_str,test_var,test_var2, test_str2, lin_var, T, sec_test_var, N_ped, fps,mot_frac

def frame_reader(folder_frame,key):
    var_frame = folder_frame[key]
    return var_frame

def label_var(test_var):
    if test_var == 0:
        return "o = "
    if test_var == 1:
        return "b = "
    if test_var == 2:
        return "v0 = "
    if test_var == 3:
        return "T = "
    if test_var == 4:
        return "rho = "
    if test_var == 5:
        return "N_ped = "
def title_var(test_var):
    if test_var == 0:
        return "variance of noise"
    if test_var == 1:
        return "corridor width [m]"
    if test_var == 2:
        return "desired velocity [m/s]"
    if test_var == 3:
        return "slope of velocity function [s]"
    if test_var == 4:
        return "initial density in [m^-2] "
    if test_var == 5:
        return "number of agents"

def error_plot_writer(dens_files,name,t_min,interval,fps_step):
    path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps,mot_frac = var_ini()
    k = 0
    array_shape = lin_var[test_var2].shape[0] * lin_var[test_var].shape[0]
    mean_array = np.empty(array_shape)
    errup_array = np.empty(array_shape)
    errdown_array = np.empty(array_shape)
    x_array = np.empty(array_shape)
    col_array = np.empty(array_shape).astype(np.str)

    t_max = int(t_min + interval)

    for tv2, l, m in zip(lin_var[test_var2], range(lin_var[test_var2].shape[0]),
                         np.arange(0, array_shape, lin_var[test_var].shape[0])):
        flat_list = np.empty((lin_var[test_var].shape[0], N_runs))

        # dens_new_list = np.empty((lin_var[test_var].shape[0],N_runs,shape))
        bi_count = 0
        j = 0
        dens_file_list = dens_files[dens_files[test_str2] == tv2]
        # files = dens_file_list["files"].values
        for bi in lin_var[test_var]:  # [1.2,2.0,2.3,3.4,4.5,5.6]:
            #print(bi)
            dens_list = []
            files = dens_file_list[dens_file_list[test_str] == bi]["files"].values
            #print(files)
            for fname in files:
                data = pd.read_csv(fname)
                keys = data.keys().values[1:]
                for key in keys:
                    dens_list.append(data[key])
            #print(np.array(dens_list).shape)
            density_np = np.array(dens_list)
            #print("dens_list = ", density_np)
            dens_mean = np.empty(density_np.shape[0])
            i = 0
            for d in density_np:
                dens_mean[i] = d[int(t_min / fps_step): int(t_max / fps_step)].mean()
                i += 1
            #print(dens_mean.shape)
            #print(flat_list.shape)
            flat_list[j] = dens_mean
            j += 1
        #density_reduced = flat_list
        cat = np.array([np.array([i for k in den]) for i, den in zip(np.arange(0, lin_var[test_var].shape[0]), flat_list)])

        df = pd.DataFrame()
        df['category'] = cat.flatten()
        df['density'] = flat_list.flatten()

        mean = df.groupby('category')['density'].mean()
        p025 = df.groupby('category')['density'].quantile(0.025)
        p975 = df.groupby('category')['density'].quantile(0.975)
        plt.xlabel(title_var(test_var))
        plt.ylabel('density [m^-2]')

        x = lin_var[test_var]  # np.array([var[test_var] for var in cross_var])
        #print("x = ", x)
        if test_var == 1:
            x = 2 * x
            # x = [1.2,2.0,2.3,3.4,4.5,5.6]
        err = np.array([mean - p025, p975 - mean])
        #print("x,mean=", x, mean)
        var2 = lin_var[test_var2][k]
        k += 1
        # plt.errorbar(x, mean, yerr=err, fmt='v', label=af.label_var(test_var2) + str(var2), ms=10)
        #print(mean_array.shape)
        #print(mean_array[m:m + lin_var[test_var].shape[0]].shape)
        #print(col_array[m:m + lin_var[test_var].shape[0]].shape)
        col_array[m:m + lin_var[test_var].shape[0]].fill(str(tv2))
        mean_array[m:m + lin_var[test_var].shape[0]] = mean
        errup_array[m:m + lin_var[test_var].shape[0]] = err[0]
        errdown_array[m:m + lin_var[test_var].shape[0]] = err[1]
        x_array[m:m + lin_var[test_var].shape[0]] = x

    #print(x_array.shape, mean_array.shape, errup_array.shape, errdown_array.shape, col_array.shape)

    df_error = pd.DataFrame(
        {test_str: x_array, "mean": mean_array, "errorup": errup_array, "errordown": errdown_array, "t": col_array})
    df_error.to_csv(path + "/density/" + name)

    # plt.show()
    #print(col_array)
    #print(df_error.head)
