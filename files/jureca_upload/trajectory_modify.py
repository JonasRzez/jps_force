#!/usr/bin/env python
# coding: utf-8
# In[23]:
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import math as m
import os
from multiprocessing import Pool
import fileinput
#import random as rd
#from pathlib import Path
#import sys
#from scipy.integrate import simps
import itertools

from shapely.geometry import Point, Polygon

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def room_geo(b):
    b = b/2
    #coords = [(-0.25,-1), (-0.25,-0.15),(-0.4,0),(-b/2,0),(-b/2,5),(b/2,5),(b/2,0),(0.4,0), (0.25,-0.15),(0.25,-1)]
    coords = [(0,0),(-b,0),(-b,5),(b,5),(b,0),(0.0,0)]

    poly_room = Polygon(coords)
    return poly_room

def wallbuilder(b):
    b = b/2
    wall_lx = np.empty(100)
    wall_lx.fill(-b)
    wall_ly = np.linspace(0,7,100)
    wall_l = np.array([np.array([xi,yi]) for xi,yi in zip(wall_lx,wall_ly)])

    wall_rx = np.empty(100)
    wall_rx.fill(b)

    wall_bx = np.array([i for i in np.arange(-b,b,0.1) if abs(i) > 0.25])
    wall_by = np.empty(wall_bx.shape)
    wall_by.fill(0)
    wall_b = np.array([np.array([xi,yi]) for xi,yi in zip(wall_bx,wall_by)])
    wall_ry = np.linspace(0,7,100)
    wall_r = np.array([np.array([xi,yi]) for xi,yi in zip(wall_rx,wall_ry)])
    wall = np.vstack((wall_l,wall_r))
    wall = np.vstack((wall,wall_b))
    return wall

def vor_dens(lat_x,lat_y,poly_room,wall):
    lat = np.array([np.array([xi, yi]) for xi, yi in zip(lat_x, lat_y)])
    measure_poly = Polygon([(-0.4,0.5),(-0.4,1.3),(0.4,1.3),(0.4,0.5)])

    lat = np.array([point for point in lat if Point(point[0], point[1]).within(poly_room)])
    if lat.shape[0] == 0:
        return 0
    lat = np.vstack((lat,wall))

    vor = Voronoi(lat)
    vert = vor.regions
    rig_vert = []
    for note in vert:
        if -1 in note:
            continue
        rig_vert.append(note)

    measure_area = measure_poly.area
    #densty = []
    #densty = np.empty(np.array(rig_vert[1:]).shape[0])
    #count = 0
    dens = 0
    for note in rig_vert[1:]:
        coords = [(vor.vertices[i][0],vor.vertices[i][1]) for i in note]
        poly = Polygon(coords)
        intersec = poly.intersection(poly_room)
        if intersec.is_empty:
            continue
        poly = intersec
        pol_area = poly.area
        solution = measure_poly.intersection(poly)
        if pol_area > 0:
            #densty.append(solution.area * 1/pol_area)
            #densty[count] = solution.area * 1/pol_area
            dens += solution.area * 1/pol_area
            #count += 1
    #return density
    #densty = np.array(densty)
    return dens/measure_area



def key_to_float(keys):
    key_list = []

    for key in keys[1:]:
        """
        comma = False
        new_key = ' '
        for l in key:
            if comma == True:
                new_key += l
            if l == ",":
                comma = True"""
        new_key = float(new_key)
        key_list.append(new_key)
    return key_list
    
def csv_writer(file_name,path,df,append):
    location = path + file_name
    if os.path.isfile(location) and append:
        load_frame = pd.read_csv(location)
        key_list = load_frame.keys()[1:]
       # key_list = key_to_float(keys)
        df_keys = df.keys()
        #df_keys = key_to_float(df.keys())
        keys_float = [float(key) for key in key_list]
        for key in df_keys:
            keys_float.append(float(key))
        print("key float " , keys_float)
        keys_float.sort()
        for old_key in key_list:
            for new_key in df_keys:
                if old_key == new_key:
                    del df[str(new_key)]
                    print("deleted key " + str(new_key))
        df_app = pd.DataFrame()
        for key in keys_float:
            key = str(key)
            if key in key_list:
                df_app[key] = load_frame[key]
            if key in df_keys:
                df_app[key] = df[key]
            else:
                print("key " + str(key) + " not found")
        df_app.to_csv(location, mode = "w", header = True)
    else:
        df.to_csv(location)


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
    #traj_angle_frame = pd.DataFrame(angle_dic)
 
    return traj_x_frame, traj_y_frame#, traj_angle_frame


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

def frame_reader(test_var,var,folder_frame,key,path):
    """
    if test_var == var or test_var2 == var:
        var_frame = folder_frame[key]
    else:
        var_frame = pd.read_csv(path)"""
    var_frame = folder_frame[key]

    return var_frame
    
def density_csv_writer(density,test_str,test_str2,var2,lin_var):
    dens_map = {}

    density1 = np.array(density)
    shape_max = 0
    for d in density1:
        #print(d.shape)
        shape = d.shape[0]
        if shape > shape_max:
            shape_max = shape
    #print(shape_max)
    count = 0
    for d in density1:
        if d.shape[0] < shape_max:
            diff = int(shape_max-d.shape[0])
            nod_list = [0 for i in np.arange(0,diff)]
            d = np.append(d,nod_list)
            density1[count] = d
        count+=1
    count = 0

    for d,var in zip(density1,lin_var[test_var]):
        dens_map[str(round(var,3))] = d
        
    dens_frame = pd.DataFrame(dens_map)
    dens_name =  'dens_mean_' + test_str + "_" +test_str2 + "_" + str(var2) + '.csv'
    csv_writer(dens_name,path,dens_frame,append)

# In[25]:



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
print("current path is " ,path )
folder_frame = pd.read_csv(path + "folder_list.csv")
"""
if test_var == 0:
    print("test variable esigma")
    folder_frame.sort_values(by = ["esigma"])
if test_var == 1:
    print("test variable b")
    folder_frame.sort_values(by = ["b"])
if test_var == 2:
    print("test variable v0")
    folder_frame.sort_values(by = ["v0"])
if test_var == 3:
    print("test variable T")
    folder_frame.sort_values(by = ["T"])
if test_var == 4:
    print("test variable rho")
    folder_frame.sort_values(by = ["rho"])
"""

folder_list = np.array(folder_frame['ini_folder'])

folder = "trajectories/"



#rho_ini_frame = pd.read_csv(path + "rho_ini.csv")
#rho_ini = np.array(rho_ini_frame['rho_ini'])

testvar_list = pd.read_csv(path + test_str + "_list.csv")
testvar_list = np.array(testvar_list)
test_var_shape = testvar_list.shape[0]

esigma_frame = frame_reader(test_var,0,folder_frame,"esigma",path + "esigma_list.csv")
esigma = np.array(esigma_frame)

T_frame = frame_reader(test_var,3,folder_frame,"T",path + "T_list.csv")
print(T_frame)
T = np.array(T_frame)
print(T)
v0_frame = frame_reader(test_var,2,folder_frame,"v0",path + "v0_list.csv")
v0 = np.array(v0_frame)

rho_ini_frame = frame_reader(test_var,4,folder_frame,"rho",path + "rho_list.csv")
rho_ini = np.array(rho_ini_frame)

b_frame = frame_reader(test_var,1,folder_frame,"b",path + "b_list.csv")
b = np.array(b_frame)

N_runs = variables['N_runs'][0]
fps = variables['fps'][0]
b = np.array(b_frame)
N_ped = variables['N_ped'][0]
t_max = variables['t_max'][0]


#cross_var = Product(np.array([esigma,b,v0,T,rho_ini]))
cross_var = np.load(path + "cross_var.npy")
lin_var = np.load(path + "var.npy",allow_pickle=True)
print(cross_var)
print(folder_frame)
print(rho_ini)

print("b/2 = ", b)
print("fps = " ,fps)
print("N_runs = ", N_runs)
print("folder_list_0", folder_list[0])
print("folder_list_-1", folder_list[-1])
print("N_ped = ", N_ped)
print("t_max = ", t_max)

    
print("writing files")
folder_anal_frame = [np.array(folder_frame[(folder_frame[test_str] == var[test_var])]['ini_folder'])[0] for var in cross_var]
print(folder_anal_frame)


for i in range(N_runs):
    #location = np.array([["evac_traj_" +b_data_name(2 * bi,3) + "_"+ str(i) +".txt" for bi in b] for esig in esigma])
    location = np.array(["evac_traj_" +b_data_name(2 * b[var],3) + "_"+ str(i) +".txt" for var in range(cross_var.shape[0])])
    #location = location.flat
    for loc,run_folder in zip(location, folder_list):

        file_name = path  + run_folder + sl +loc
        if os.path.isfile(file_name):
            file = open(file_name, 'r')
            line_count = 0
            new_file = open(path  + run_folder+ sl + "new_" + loc ,'w')
            #print(loc)
            for line in file:
                line_count += 1
                # if line_count > 10:
                if line_count == 13:
                    new_file.write(line[1:])
                else:
                    new_file.write(line)

            os.system("rm " + path  + run_folder + sl +loc)
            new_file.close()
        else:
            print("WARNING: file " + file_name + " not found")
print("/writing files")


dens_folder = "density_runs"
dens_folder_run = "density_" + str(N_ped) + "_" + str(t_max)
os.system('mkdir ' + path +dens_folder)
os.system('pwd')
count = 0
for var in lin_var[test_var2]:
    density_mean = []
    density = []
    print(var)
    folder_frame_frac = folder_frame.loc[folder_frame[test_str2] == var]
    b_frac = np.array(folder_frame_frac['b'])
    print("linvar.shape",lin_var[test_var].shape[0])
    for j, ini in zip(range(lin_var[test_var].shape[0]), np.array(folder_frame_frac['ini_folder'])):
        location = "new_evac_traj_" + b_data_name(2 * b_frac[j],3)
        density_runs = []
        max_frame = 0
        for i in range(N_runs):
            loc = path + ini + sl + location + "_" + str(i) + ".txt"
            #print(loc)
            if(os.path.isfile(loc)):
                if os.stat(loc).st_size == 0:
                    print("found empty file")
                    continue
                traj = pd.read_csv(loc, sep="\s+", header=0, comment = '#',index_col=False)
                frame = max_frame_fetch(traj)
                if frame > max_frame:
                    max_frame = frame
        
        density_run = []
        density_runs_map = {}
        wall = wallbuilder(2 * b_frac[j])
        room = room_geo(2 * b_frac[j])
        for i in range(N_runs):
            loc = path  + ini + sl + location + "_" + str(i) + ".txt"
            print(loc)
            if(os.path.isfile(loc)):
                if os.stat(loc).st_size == 0:
                    print("WARING: found empty file " + loc)
                    continue

                """if traj.empty:
                    print("empty tajectoy file")
                continue"""
                traj = pd.read_csv(loc, sep="\s+", header=0, comment = "#")
                l_x, l_y = lattice_data(traj)
                
                fwhm = 0.2
                a = fwhm * m.sqrt(2) / (2 * m.sqrt(2 * m.log(2)))
                lattice_x = np.array(l_x)
                lattice_y = np.array(l_y)
                x_array = np.linspace(-0.5,0.5,25)
                y_array = np.linspace(0.5,1.5,25)
                print("*****************<calc density>*****************")
                lat_x_no_nan = []
                lat_y_no_nan = []

                for lat_x,lat_y in zip(lattice_x,lattice_y):
                    l_x_no_nan = [x  for x in lat_x if np.isnan(x) == False]
                    l_y_no_nan = [y  for y in lat_y if np.isnan(y) == False]
                    lat_x_no_nan.append(l_x_no_nan)
                    lat_y_no_nan.append(l_y_no_nan)

                print("    *****************<pool>*********************")
                pool = Pool()
                g_pool = np.array([pool.apply_async(vor_dens, args=(l_x_no_nan,l_y_no_nan,wall,room)) for l_x_no_nan, l_y_no_nan in zip(lat_x_no_nan, lat_y_no_nan)])
                #g_pool = np.array([pool.apply_async(normal, args=(l_x_no_nan,l_y_no_nan,x_array,y_array,a)) for l_x_no_nan, l_y_no_nan in zip(lat_x_no_nan, lat_y_no_nan)])

                density_run = [p.get() for p in g_pool]
                pool.close()
                print("    *****************</pool>********************")
                print("*****************</calc density>****************")

                
                dens_run_shape = np.array(density_run).shape[0]
                print(np.array(density_run).shape)
                

                if dens_run_shape < max_frame:
                    diff = int(max_frame-dens_run_shape)
                    nod_list = [0 for i in np.arange(0,diff)]
                    density_run = np.append(density_run,nod_list)
                density_runs_map[str(i)] = density_run
                density_runs.append(np.array(density_run))
                #print(density_runs)
            else:
                print("WARNING: can not find file location of " + loc)
        if(os.path.isfile(loc)):
            if os.stat(loc).st_size > 0:
                mean_runs_dens = np.array(density_runs).mean(axis=0)
                std_runs_dens = np.array(density_runs).std(axis=0)
                density.append(mean_runs_dens)
                dens_i_df = pd.DataFrame(density_runs_map)
                dens_i_df.to_csv(path + dens_folder  + "/" +"densities_i_esigma_" + str(esigma[j]) + "_b_" + str(b[j]) + "_v0_" + str(v0[j]) + "_T_" + str(T[j]) + "_rho_" + str(rho_ini[j])  + ".csv")
                print("density shape = ", np.array(density).shape)
        print(count)
        
    print("shape = ", np.array(density).shape)
    density_csv_writer(density,test_str,test_str2,lin_var[test_var2][count],lin_var)
    count += 1



# In[41]:


#print(std_runs_dens)


print("programm finished with no errors")
