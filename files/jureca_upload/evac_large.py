#!/usr/bin/env python
import os
import numpy as np
import itertools
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import random as rand
PATH = os.path.dirname(os.path.abspath("evac_geo_temp.xml"))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(PATH),
    trim_blocks=False)

def csv_writer(file_name,path,df,key,append):
    location = path + file_name
    if os.path.isfile(location) and append:
        load_frame = pd.read_csv(location)
        b_frame = load_frame[key]
        b_np = np.array(b_frame)
        for bi in b_np:
            df = df[df[key] != bi]
        df.to_csv(location, mode = "a", header = False)
    else:
        df.to_csv(location)

def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)

def Product(variables):
    return list(itertools.product(*variables))

def create_inifile(geo_name, geo_name_ini, cross_new, ini_list, ini_folder_list , location, folder, r,  stepsize, fps, l0_list, t_max,periodic, rand_mot, mot_hm_lm,ini_i):
    
   #location = [path + "evac_traj_" + str(2 * bi)[0] + str(2*bi)[-1] + ".txt" for bi in b_list]
    #print(geo_name,location)

    for var,fname,l0 in zip(cross_new, geo_name,l0_list):
        context= {'b': var[1],'l':l0, 'll': 1.5 * l0,"wedge":var[1] - 0.4}
        fname = folder + fname
        if os.path.isfile(fname) == False:
            #print("output geo: ",  fname)
            print(os.system("pwd"))
            with open(fname,'w') as f:
                xml = render_template('evac_geo_temp.xml', context)
                f.write(xml)
    #print("shapes " , np.array(geo_name_ini).shape,np.array(location).shape,np.array(ini_list).shape,np.array(cross_new).shape)
    
    for geo,loc,fname_ini,var,l0,ini_folder_i in zip(geo_name_ini,location,ini_list,cross_new,l0_list,ini_folder_list):
        print("ini_folder_i = ", ini_folder_i)
        print("fname iini = " , fname_ini)
        print("folder = ", folder)
        seed = int(rand.uniform(0, 1485583981))
        if rand_mot:
            ini_context = {'geo':geo,'location':loc,'b':var[1], 'r':r,'l':l0, 'll':  1.5 * l0,'seed':seed,'stepsize':stepsize,'fps':fps,'N_ped_lm':int(var[5] * (1 - var[6])),'N_ped_hm': int(var[5] * var[6]),'esig':var[0],'t_max':t_max,'periodic':periodic,'v0':var[2],'T_lm':mot_hm_lm[1],'T_hm':mot_hm_lm[0], "aviod_wall":var[7],'output': "../../files/jureca_upload/" + folder + ini_folder_i }
        else:
            ini_context = {'geo':geo,'location':loc,'b':var[1], 'r':r,'l':l0, 'll':  1.5 * l0,'seed':seed,'stepsize':stepsize,'fps':fps,'N_ped_lm':int(var[5] * (1 - var[6])),'N_ped_hm': int(var[5] * var[6]),'esig':var[0],'t_max':t_max,'periodic':periodic,'v0':var[2],'T_lm':var[3],'T_hm':var[3], "aviod_wall":var[7],'output': "../../files/jureca_upload/" + folder + ini_folder_i}

        #l = 0
        #l_add = round(l0/6,2)
        """for i in range(1,7): #devides the room into 6 slices to distribute pedestrians
            l += l_add
            ini_context["l" + str(i)] = l"""
        #print("ini_context = ", ini_context)
        fname_ini = folder + fname_ini
        #print("output ini: ", fname_ini)
        with open(fname_ini, 'w') as f:
            xml_ini = render_template('evac_ini_temp_diff.xml', ini_context)
            f.write(xml_ini)
            
def b_data_name(b,dig):
    b = round(b,dig)
    str_b = ''
    for let in str(b):
        if (let in '.'):
            str_b += '_'
        else:
            str_b += let
    return str_b
    
def test_var_fetch(test_var):
    if test_var == "mot_frac":
        var_i = 6
    if test_var == "N_ped":
        var_i = 5
    if test_var == "rho":
        var_i = 4
    if test_var == "T":
        var_i = 3
    if test_var == "v0":
        var_i = 2
    if test_var == "b":
        var_i = 1
    if test_var == "esigma":
        var_i = 0

    else:
        print("test var not included")
    return var_i

def ini_bool():
    sec_test_var = True
    append = False
    rho_ini_rand = False
    rand_mot = False
    return sec_test_var,append,rho_ini_rand,rand_mot
def test_var_ini():
    dig = 3  # for the rounding of digits
    test_var = "esigma"
    test_var2 = "T"
    motivation = "_lm_"
    var_i = test_var_fetch(test_var)

    return dig,test_var,test_var2,var_i,motivation

def var_ini(i_start,i_end,esigma):
    #rho_min = 2.0
    #rho_max = 3.0
    rho_min = 2.0
    rho_max = 2.0
    rho_ini = np.array([(rho_min + rho_max) / 2])
    #rho_ini = np.array([0.5,0.8,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.4,2.6,2.8,3.0,4.0])
    T = np.array([1.])
    mot_frac = np.array([1.])
    avoid_wall = np.array([0.0,0.0])
    v0 = np.array([1.34])
    #esigma = np.array([0.7])
    N_ped = np.array([5000])

    r = 0.1
    fps = 1
    stepsize = 0.01
    #N_ped = 55
    N_runs = 1
    t_max = 500
    periodic = 1
    return rho_ini, T, v0, esigma, fps, stepsize, N_ped, i_start, i_end, t_max, periodic, r, N_runs, rho_min, rho_max, avoid_wall, mot_frac

def ini_cross(cross_variable,shape_var,rho_min,rho_max,rho_ini_rand):
    cross_new = np.empty([shape_var,
                          8])  # length has to be the number of variables given currently 0:esigma,1:b, 2:v0, 3:T , 4:rho_ini: 5 N_ped 6: mot_frac
    avoid_wall = np.array([0.4])
    #T = np.array([1.0, 0.9,0.8,0.7])
    #print(cross_new)
    for i in range(shape_var):
        cross_i = cross_variable[i]
        cross_i = np.append(cross_i,avoid_wall[0])
        print(cross_i.shape, cross_i)
        cross_new[i] = cross_i  # vraiable are 0:esigma, 1:b, 2:v0, 3:T, 4:rho_ini, 5:N_ped, 6:mot_frac 7:wall avoidance
    return cross_new

def ini_csv(path,N_runs,fps,N_ped,t_max,periodic,var_i,test_var,sec_test_var,test_var2,b,append,v0,T,esigma,rho_ini,ini_folder_list,data_folder,cross_new,r,variables):
    # <saving all variable information into csv files>

    variables_df = pd.DataFrame(
        {'r': [r], 'fps': [fps], 'N_runs': [N_runs], 'N_ped': [N_ped[0]], 't_max': [t_max], 'periodic': [periodic],
         'test_var': [var_i], 'test_str': [test_var]})
    if sec_test_var:
        var_j = test_var_fetch(test_var2)
        variables_df['test_var2'] = var_j
        variables_df['test_str2'] = test_var2
    variables_df.to_csv(path + "variables_list.csv", mode="w")
    b_df = pd.DataFrame({'b': b})
    csv_writer("b_list.csv", path, b_df, 'b', append)

    v_df = pd.DataFrame({'v0': v0})
    csv_writer("v0_list.csv", path, v_df, 'v0', append)

    T_df = pd.DataFrame({'T': T})
    csv_writer("T_list.csv", path, T_df, 'T', append)

    N_df = pd.DataFrame({'N_ped': N_ped})
    csv_writer("N_ped_list.csv", path, N_df, 'N_ped', append)

    esig_df = pd.DataFrame({'esigma': esigma})
    csv_writer("esigma_list.csv", path, esig_df, 'esigma', append)

    rho_df = pd.DataFrame({'rho_ini': rho_ini})
    csv_writer("rho_list.csv", path, rho_df, 'rho_ini', append)

    mot_frac_df = pd.DataFrame({'rho_ini': rho_ini})
    csv_writer("mot_frac_list.csv", path, mot_frac_df, 'mot_frac', append)

    folder_path_list = np.array([data_folder + "/" + ini for ini in ini_folder_list])

    T_list = np.array([var[3] for var in cross_new])
    esig_list = np.array([var[0] for var in cross_new])
    v0_list = np.array([var[2] for var in cross_new])
    b_list = np.array([var[1] for var in cross_new])
    rho_list = np.array([var[4] for var in cross_new])
    N_ped_list = np.array([var[5] for var in cross_new])
    mot_frac_list = np.array([var[6] for var in cross_new])

    #print("shapes of list = ", T_list.shape,esig_list.shape,v0_list.shape,rho_list.shape,b_list.shape,np.array(ini_folder_list).shape)
    folder_df = pd.DataFrame(
        {'ini_folder': ini_folder_list, 'b': b_list, 'v0': v0_list, 'T': T_list, 'rho': rho_list, 'esigma': esig_list,'N_ped':N_ped_list, 'mot_frac':mot_frac_list})

    # folder_df.to_csv( path + "folder_list.csv")
    csv_writer("folder_list.csv", path, folder_df, test_var, append)

    path_df = pd.DataFrame({'path': [path]})
    path_df.to_csv("path.csv")

    np.save(path + "cross_var.npy", cross_new)
    np.save(path + "var.npy", variables)


# </saving information>

def ini_files(b,i_start,i_end,esigma):
    sec_test_var, append, rho_ini_rand, rand_mot = ini_bool()
    dig, test_var, test_var2, var_i, motivation = test_var_ini()
    rho_ini, T, v0, esigma, fps, stepsize, N_ped, i_start, i_end, t_max, periodic, r, N_runs, rho_min, rho_max, wall_avoidance, mot_frac = var_ini(i_start,i_end,esigma)
    b = b/2

    variables = np.array([esigma, b, v0, T, rho_ini,N_ped, mot_frac])
    cross_variable = np.array(list(itertools.product(*variables)))
    shape_var = cross_variable.shape[0]

    cross_new = ini_cross(cross_variable, shape_var, rho_min, rho_max, rho_ini_rand)

    path, data_folder, traj_folder = ini_traj_folder(motivation, N_ped, t_max, r, fps, test_var)

    ini_folder_list = ini_folder_fetch(cross_new, dig, motivation, N_ped, periodic, traj_folder, data_folder, t_max)

    ini_csv(path, N_runs, fps, N_ped, t_max, periodic, var_i, test_var, sec_test_var, test_var2, b, append, v0, T,
            esigma, rho_ini, ini_folder_list, data_folder, cross_new, r, variables)

def ini_folder_fetch(cross_new,dig,motivation,N_ped,periodic,traj_folder,data_folder,t_max):
    ini_folder_list = []
    for var in cross_new:
        ini_folder_name = "ini_" + b_data_name(var[1], dig) + motivation + str(N_ped[0]) + "_esigma_" + b_data_name(var[0],
                                                                                                                 dig) + "_tmax_" + str(
            t_max) + "_periodic_" + str(periodic) + "_v0_" + b_data_name(var[2], dig) + "_T_" + b_data_name(var[3],
                                                                                                            dig) + "_rho_ini_" + b_data_name(
            var[4], dig) + "_Nped_" + b_data_name(var[5],dig) + "_motfrac_" + b_data_name(var[6],dig)
        mkdir_path = traj_folder + data_folder + "/" + ini_folder_name
        print("os path is file = ", os.path.isdir(mkdir_path))

        if 1 - os.path.isdir(mkdir_path):
            os.system("mkdir " + mkdir_path)
        ini_folder_list.append(ini_folder_name)
    return ini_folder_list

def ini_traj_folder(motivation,N_ped,t_max,r,fps,test_var):
    traj_folder = "trajectories/"
    data_folder = "ini" + motivation + "N_ped" + str(N_ped[0]) + "_tmax" + str(t_max) + "_size_" + b_data_name(r,2) + "_fps_" + str(fps) + "_testvar_" + test_var
    print_level = "--log-level [debug, info, warning, error, off]"
    if 1 - os.path.isfile(traj_folder + data_folder):
        print(traj_folder + data_folder)
        os.system("mkdir " + traj_folder + data_folder)
    path = traj_folder + data_folder + "/"

    return path,data_folder,traj_folder

def main(b,i_start,i_end,esigma,ini_i):
    sec_test_var, append, rho_ini_rand,rand_mot = ini_bool()
    dig,test_var,test_var2,var_i,motivation = test_var_ini()
    rho_ini,T , v0,esigma, fps, stepsize, N_ped, i_start, i_end, t_max, periodic, r, N_runs,rho_min,rho_max,avoid_wall,mot_frac = var_ini(i_start,i_end,esigma)
    run_total = N_runs * b.shape[0]
    b = b / 2
    mot_hm_lm = [0.1,1.3]

    variables = np.array([esigma,b,v0,T,rho_ini,N_ped,mot_frac])
    #print("variables = " ,variables)
    cross_variable = np.array(list(itertools.product(*variables)))
    shape_var = cross_variable.shape[0]
    print("shape_var = ", shape_var)
    #print("corr variable = ", cross_variable)
    cross_new = ini_cross(cross_variable, shape_var, rho_min, rho_max, rho_ini_rand)

    path, data_folder, traj_folder = ini_traj_folder(motivation,N_ped,t_max,r,fps,test_var)

    ini_folder_list = ini_folder_fetch(cross_new, dig, motivation, N_ped, periodic, traj_folder, data_folder, t_max)
    #ini_folder_list = pd.read_csv(path + "folder_list.csv")["ini_folder"].values
    geo_name = [data_folder + "/" + ini_folder + "/" + "geo_" + b_data_name(var[1],dig) + ".xml" for var, ini_folder in zip(cross_new, ini_folder_list)]
    geo_name_ini = ["geo_" + b_data_name(var[1],dig) + ".xml" for var in cross_new]

    ini_list = [data_folder + "/" + ini_folder + "/" + "ini_" + b_data_name(var[1],dig)+ str(ini_i) + ".xml" for var, ini_folder in zip(cross_new, ini_folder_list)]
    output_list = [data_folder + "/" + ini_folder for var, ini_folder in zip(cross_new, ini_folder_list)]
    for i in range(i_start,i_end):
        print("iteration = " , i)
        location = ["evac_traj_" + b_data_name(2 * var[1],dig) + "_" + str(i) + ".txt" for var, folder_name in zip(cross_new,ini_folder_list)]
        #print("location = ", location)
        #rho_ini_rand = [rand.uniform(3,4) for i in cross_new]
        if rho_ini_rand:
            for i in range(shape_var):
                cross_new[i][4] = rand.uniform(rho_min, rho_max)

        l0_list = [round(var[5]/(2 * var[4]*var[1]),2) for var in cross_new]
        l0_list = [l0 if l0 > 7. else 7. for l0 in l0_list]
        create_inifile(geo_name,geo_name_ini,cross_new,ini_list ,output_list,location,traj_folder,r,stepsize,fps,l0_list,t_max,periodic,rand_mot,mot_hm_lm,ini_i)
        os.system("pwd")
        os.chdir("../../build/bin")
        run_count = 0
        
        for ini, ini_folder in zip(ini_list,ini_folder_list):
            jps = "./jpscore ../../files/jureca_upload/" + traj_folder + ini
            #print(jps)
            os.system(jps)
            run_count += 1
        os.chdir("../../files/jureca_upload")

if __name__ == "__main__":
    main(b,i_start,i_end,esigma,ini_i)


