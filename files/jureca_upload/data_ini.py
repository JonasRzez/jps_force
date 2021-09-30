import pandas as pd
import AnalFunctions as af
import numpy as np
import matplotlib.pyplot as plt
import os
import math as m
import seaborn as sns
from scipy.spatial import Delaunay
import CorFunFast as cff
import scipy.integrate
from multiprocessing import Pool
import networkx as nx
import StructureFunctions as sf
from itertools import groupby

def dataFrameInitilizer(load,li,lf,index_bool,ind,time_list):
    
    path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()
    af.file_writer(path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var)

    sl = "/"
    T_test_list = lin_var[test_var2]
    lattice_type = 'jule'
    runs_tested = N_runs
    traj_testvar2 = []

    sf.folderBuilder(path)
    x_min = 7
    x_max = -7
    y_min = 0
    y_max = 14
    box= [x_min,x_max,y_min,y_max]
    x_l = []
    y_l = []
    col = ["ID","FR","X","Y","speed_nn","COLOR","ANGLE"]
    #col = ["id" ,"frame", "x/cm", "y/cm"]
    sig = 3.

    t2 = T_test_list[-1]

    T_test_list = [T_test_list[-1]]
    esigmas = lin_var[test_var][li:lf]
    print("esigmas = ", esigmas)

    blist = 2 * lin_var[test_var]
    filtered = True
    XYFileSystem = []
    #time_list = np.arange(0,900,1)
    time_list = sorted(np.unique(time_list))

    for T_test in T_test_list:
        bi = li
        loc_list = sf.folderCollector(folder_frame,T_test)
        loc_list = loc_list[li:lf]
        for loc_list_runs in loc_list:
            print("<calculating " + test_str + " = " + str(lin_var[test_var][bi]) + ">")
            print(len(loc_list_runs))
            df_list = []
            for loc, ni in zip(loc_list_runs,range(len(loc_list_runs))):
                if sf.filechecker(loc) and load:
                    continue
                print("ni = ", ni)
                df_list.append(sf.fileload(loc,load,col))
                    
            
            for second in time_list:
                print("second = " , second)
                for df, ni in zip(df_list,range(len(df_list))):
                    csvname = path + "plots/structure/XYcsv/" + af.b_data_name(lin_var[test_var][bi],3) + "t" + str(second) + "runi_" + str(ni) + test_str2+ str(T_test) + ".csv"
                    print(csvname)
                    if os.path.isfile(csvname):
                        print("file exists")
                        df_read = pd.read_csv(csvname)
                        box = [df_read['x'].min() - 0.1,df_read['x'].max() + 0.1,df_read['y'].min() - 0.1,df_read['y'].max() - 0.1]
                        keys = df_read.keys()
                        if 'id' in keys:
                            print("id is in keys")
                        else:
                            sf.orderfetch(second,ni,df,fps,test_str2,T_test,csvname,True,box)
                        
                    else:
                        if load == False:
                            continue
                        print("Calculate Order")
                        sf.orderfetch(second ,ni,df,fps,test_str2,T_test,csvname,True,box)
                    XYFileSystem.append([csvname,second,ni,lin_var[test_var][bi],T_test])
            bi += 1
            
    dfcsv = pd.DataFrame()
    dfcsv['files'] = [i[0] for i in XYFileSystem]
    dfcsv['time'] = [i[1] for i in XYFileSystem]
    dfcsv['index'] = [i[2] for i in XYFileSystem]
    dfcsv['testvar'] = [i[3] for i in XYFileSystem]
    dfcsv[test_str2] = [i[4] for i in XYFileSystem]
    if index_bool:
        dfcsv = dfcsv[dfcsv['index'].isin(ind)]
    dfcsv.to_csv(path + "plots/structure/XYcsv/filelist.csv")
    
    

    df = sf.DataFrameBuilder2(esigmas,T_test_list)
    
    return df, df_list
