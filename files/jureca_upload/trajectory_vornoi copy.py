
def voronoi_density(test_i,test_f):
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
    import AnalFunctions as af
    from shapely.geometry import Point, Polygon
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    def exp_plot(exp_dens,mot,c):
        b22 = [1.2,2.3,3.4,4.5,4.5,5.6,5.6]

        p1 = np.array(exp_dens['p025'])
        p2 = np.array(exp_dens['p975'])
        mean_exp =  np.array(exp_dens['mean'])
        plt.errorbar(b22,mean_exp,yerr =[mean_exp - p1, p2 - mean_exp], label = "Experimental results " + mot, color = c,fmt='o')

    def wallbuilder(b):
        wall_lx = np.empty(100)
        wall_lx.fill(-2 * b)
        wall_ly = np.linspace(0, 15, 100)
        wall_l = np.array([np.array([xi, yi]) for xi, yi in zip(wall_lx, wall_ly)])

        wall_rx = np.empty(100)
        wall_rx.fill(2 * b)

        wall_bx = np.array([i for i in np.arange(-b, b, 0.1) if abs(i) > 0.25])
        wall_by = np.empty(wall_bx.shape)
        wall_by.fill(-4)
        wall_b = np.array([np.array([xi, yi]) for xi, yi in zip(wall_bx, wall_by)])
        wall_ry = np.linspace(0, 15, 100)
        wall_r = np.array([np.array([xi, yi]) for xi, yi in zip(wall_rx, wall_ry)])
        wall = np.vstack((wall_l, wall_r))
        wall = np.vstack((wall, wall_b))
        return wall

    def room_geo(b):
        #coords = [(-0.25,-1), (-0.25,-0.15),(-0.4,0),(-b/2,0),(-b/2,5),(b/2,5),(b/2,0),(0.4,0), (0.25,-0.15),(0.25,-1)]
        coords = [(0, 0), (-b / 2, 0), (-b / 2, 15), (b / 2, 15), (b / 2, 0), (0.0, 0)]

        poly_room = Polygon(coords)
        return poly_room

    def measure_area(x_min,x_max,y_min,y_max):
        measure_poly = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
        return measure_poly

    def vor_dens(lat, poly_room, wall,measure_poly):

        lat = np.array([point for point in lat if Point(point[0], point[1]).within(poly_room)])
        if lat.shape[0] == 0:
            return
        lat = np.vstack((lat, wall))
        vor = Voronoi(lat)
        vert = vor.regions
        rig_vert = []
        for note in vert:
            if -1 in note:
                continue
            rig_vert.append(note)

        measure_area = measure_poly.area
        densty = []

        for note in rig_vert[1:]:
            coords = [(vor.vertices[i][0], vor.vertices[i][1]) for i in note]
            poly = Polygon(coords)
            intersec = poly.intersection(poly_room)
            if intersec.is_empty:
                continue
            poly = intersec
            pol_area = poly.area
            solution = measure_poly.intersection(poly)
            if pol_area > 0:
                densty.append(solution.area * 1 / pol_area)
        densty = np.array(densty)

        return densty.sum() / measure_area




    path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()
    af.file_writer(path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var)


    file_path = "../../expdata/"
    sl = "/"
    os.system("mkdir " + path + "plots/heatmaps")
    os.system("mkdir " + path + "density")

    lattice_type = 'jule'
    traj_testvar2 = []
    print(test_str2 + " " , lin_var[test_var2])
    runs_tested = N_runs
    col = ["FR","X","Y"]

    test_array = lin_var[test_var2]
    if test_f > test_array.shape[0]:
        print("WARNING: test_f outside of test_array")
    test_slice = test_array[test_i:test_f]
    for T_test in test_slice:
        print(T_test)
        folder_frame_frac = folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'].to_numpy()
        b_folder = folder_frame.loc[folder_frame[test_str2] == T_test]['b'].to_numpy()

        """if test_str2 != 'b':
            b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])
        else:
            b_folder = T_test"""
        print(b_folder)

        loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi,3) + "_" + str(i) + ".txt" for i in range(runs_tested)] for folder,bi in zip(folder_frame_frac,b_folder)]
        print(np.array(loc_list).shape)
        print("load trajectories " + str(T_test))
        trajectory_frame = [[loc for loc in loc_list_runs if os.path.isfile(loc)] for loc_list_runs in loc_list]
        #print(trajectory_frame)
        traj_testvar2.append(trajectory_frame)
        print("/load trajectories")
    fps = 16
    fps_step = 8
    k = 0
    #shape = 50
    dens_name = []
    dens_name_ini = []
    testvar2_list = []
    testvar_list = []
    #b_list_list = [[0.6,0.6,0.6,0.6],[1.15,1.15,1.15,1.15],[1.7,1.7,1.7,1.7],[2.8,2.8,2.8,2.8]]
    #b_list_list = np.array([[0.6,0.6],[2.8,2.8]])
    test_var_shape = lin_var[test_var].shape[0]
    #b_np = np.empty(test_var_shape)
    #b_list_list = np.empty()
    if test_str2 == 'b':
        b_list_list = np.array([test_slice for i in range(test_var_shape)])
    if test_str == 'b':
        b_list_list = np.array([b_folder for i in range(test_slice.shape[0])])
    else:
        b_list_list = np.array([np.full(test_var_shape, test_slice[0]) for i in range(test_var_shape)])
    #b_list_list = [b_folder]
    print("b_folder = ", b_folder)
    min_t = int(fps * 10)
    max_t = int(min_t + fps * 5)
    
    for trajectory_frame,b_folder in zip(traj_testvar2,b_list_list):
        flat_list = np.empty((lin_var[test_var].shape[0],N_runs))
        #dens_new_list = np.empty((lin_var[1].shape[0],N_runs,shape))
        print(flat_list.shape)
        #bi_count = 0
        vor_mean01 = []
        j = 0
        print(b_folder)
        for loc_run,bi in zip(trajectory_frame,2 * b_folder):
            #bi = 2 * lin_var[1][bi_count]
            #bi_count += 1
            N_sim = len(loc_run)
            density_list = []
            density_list_ini = []
            df_dens = pd.DataFrame()
            df_dens_ini = pd.DataFrame()

            print(test_str, " = ",str(lin_var[test_var][j]), test_str2 , " = ",str(test_slice[k]) )
            measure_poly = measure_area(-0.4, 0.4, 0.5, 1.3)
            print("*****************<calc density>*****************")
            for loc in loc_run:
                if os.stat(loc).st_size == 0:
                    print("WARNING: file" + loc + " is empty")
                    continue
                traj_i = pd.read_csv(loc, sep="\s+", header=0, comment = "#",usecols = col)
                wall = wallbuilder(bi)
                room = room_geo(bi)
                densty = []
                density_ini = []
                frame_array = np.arange(0, traj_i['FR'].max(), fps_step)
                
               
                    
                
                for i in frame_array:
                    x_list = np.array(traj_i[traj_i['FR'] == i]['X'])
                    y_list = np.array(traj_i[traj_i['FR'] == i]['Y'])
                    lattice = np.array([np.array([xi, yi]) for xi, yi in zip(x_list, y_list)])
                    if int(i) == 0:
                        print(bi)
                        d_ini = vor_dens(lattice,room,wall,measure_area(-bi/2,bi/2,0,7))
                        print(d_ini)
                        density_ini.append(d_ini)
                    di = vor_dens(lattice, room, wall, measure_poly)
                    densty.append(di)
                density_list.append(np.array(densty))
                density_list_ini.append(np.array(density_ini))
                t_array = [i / fps for i in frame_array]
            print("*****************</calc density>*****************")

            # plt.show()
            density_np = np.array(density_list)
            density_np_ini = np.array(density_list_ini)
            dens_length = np.empty((N_sim))
            #print(density_np.shape)
            for d,i in zip(density_np,range(density_np.shape[0])):
                dens_length[i] = d.shape[0]
            shape = int(dens_length.min())
            i = 0
            #dens_new = np.empty((N_runs, shape))

            dens_mean = np.empty(density_np.shape[0])
            dens_mean_ini = np.empty(density_np_ini.shape[0])
            #print(shape)
            for d,i in zip(density_np,range(density_np.shape[0])):
                dens_mean[i] = d[int(min_t / fps_step): int(max_t / fps_step)].mean()
                #i += 1
                df_dens[str(i)] = d[0:shape]
            for d,i in zip(density_np_ini,range(density_np_ini.shape[0])):
                dens_mean_ini[i] = d.mean()
                df_dens_ini[str(i)] = d[0:shape]

            #print("mean density = ", dens_mean.mean())
            #print("std denstiy = ", dens_mean.std())
            dens_mean_shape = dens_mean.shape[0]
            if dens_mean_shape < N_runs:
                print("WARNING: Fitting shape")
                diff_shape = N_runs - dens_mean_shape
                dens_mean = np.append(dens_mean,np.zeros(diff_shape))
            flat_list[j] = dens_mean
            #print(dens_new_list.shape,dens_new.shape)
            #dens_new_list[j] = dens_new
            dens_file_name = path + "density/" + "dens_" + test_str2 + "_" + str(test_slice[k]) + "_" + test_str + "_" + str(lin_var[test_var][j]) + ".csv"
            dens_file_name_ini = path + "density/" + "dens_ini_" + test_str2 + "_" + str(test_slice[k]) + "_" + test_str + "_" + str(lin_var[test_var][j]) + ".csv"

            df_dens.to_csv(dens_file_name)
            df_dens_ini.to_csv(dens_file_name_ini)

            dens_name.append(dens_file_name)
            dens_name_ini.append(dens_file_name_ini)

            testvar2_list.append(test_slice[k])
            testvar_list.append(lin_var[test_var][j])
            j += 1

        density_reduced = flat_list
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
        plt.ylabel('density [m^-2]')

        #x = var_new  # np.array([var[test_var] for var in cross_var])
        if test_var == 1:
            x = 2 * lin_var[1]

        err = [mean - p025, p975 - mean]
        #print("x,mean=", x, mean)
        var2 = test_slice[k]
        k += 1
        #plt.errorbar(x, mean, yerr=err, fmt='v', label= af.label_var(test_var2) + str(var2), ms=10)
    df_densfiles = pd.DataFrame()
    df_densfiles["files"] = dens_name
    df_densfiles[test_str] = testvar_list
    df_densfiles[test_str2] = testvar2_list
 
    df_densfiles_ini = pd.DataFrame()
    df_densfiles_ini["files"] = dens_name_ini
    df_densfiles_ini[test_str] = testvar_list
    df_densfiles_ini[test_str2] = testvar2_list
    dens_files_path = path +"density/file_list.csv"
    dens_files_ini_path = path +"density/file_list_ini.csv"
    if os.path.isfile(dens_files_path):
        df_densfiles.to_csv(path +"density/file_list.csv",mode = "a",header = 0)
    else:
        df_densfiles.to_csv(path +"density/file_list.csv",mode = "w")
        
    if os.path.isfile(dens_files_ini_path):
        df_densfiles_ini.to_csv(path +"density/file_list_ini.csv",mode = "a",header = 0)
    else:
        df_densfiles_ini.to_csv(path +"density/file_list_ini.csv",mode = "w")

    #dens_files = pd.read_csv(path + "/density/file_list.csv")
    #print(dens_files)
    #dens_files_ini = pd.read_csv(path + "/density/file_list_ini.csv")
    #af.error_plot_writer(dens_files,"error_plot.csv",10 * fps,5 * fps,fps_step)
    #af.error_plot_writer(dens_files,"error_plot5s.csv",5 * fps,5 * fps,fps_step)

    #af.error_plot_writer(dens_files_ini,"error_plot_ini.csv",0,fps_step,fps_step)



