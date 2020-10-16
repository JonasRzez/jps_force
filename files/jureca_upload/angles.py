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

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    #u_new = np.array([1 if ui > 1 else 0 for ui in u])
    u[u > 1] = 1
    u[u < 0] = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = np.sqrt(dx*dx + dy*dy)
    return dist

os.system("mkdir " + path  + "distributions")
col = ["FR", "X", "Y", "ID", "speed_nn", "ANGLE_int_nn","IntID"]
blist = 2 * lin_var[test_var]
v_0_list = []
resultfolder = "distributions/"
os.system("mkdir " + path + resultfolder)
os.system("mkdir " + path + "plots")
os.system("mkdir " + path + "plots/angledist")
os.system("mkdir " + path + "plots/speeddist")

folderlist = []
for T_test in T_test_list:
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])

    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(runs_tested)] for folder, bi in zip(folder_frame_frac, b_folder)]
    #loc_list = [["ini_2_9_lm_55_esigma_0_7_tmax_156_periodic_0_v0_1_34_T_1_3_rho_ini_3_6_Nped_55_0_motfrac_1_0" + sl + "new_evac_traj_" + af.b_data_name( 5.80000001, 3) + "_" + str(i) + ".txt" for i in
     #            range(900)]]
    bi = 0

    v_0_mean = np.empty(blist.shape[0])
    v_0_var = np.empty(blist.shape[0])
    angle_nn_var = np.empty(blist.shape[0])
    angle_nn_mean = np.empty(blist.shape[0])
    for loc_list_runs in loc_list:
        print("<calculating " + test_str + " = " + str(2 * lin_var[test_var][bi]) + ">")
        v_nn_mean = np.empty(0)
        angle_nn = np.empty(0)
        p0count = 0
        for loc in loc_list_runs:
            if os.path.isfile(loc) == False:
                print("WARNING: file " + loc + " not found.")
                continue
            if os.stat(loc).st_size == 0:
                print("WARNING: file" + loc + " is empty")
                continue
            df = pd.read_csv(loc, sep="\s+", header=0, comment="#", skipinitialspace=True, usecols=col)
            #print(loc)
            # df['ANGLE_int_nn'] = df['ANGLE_int_nn'].values.astype(np.float)
            df = df[df['FR'] > 10 * fps]
            df = df[df['IntID'] > 0]
            df = df[df['Y'] > 0]

            df = df[df['ANGLE_int_nn'].values.astype(np.float) >= -1.0]
            #df = df[df['ANGLE_int_nn'] != 1.]
            #df = df[df['ANGLE_int_nn'] != 0.]
            df = df[df['speed_nn'].values.astype(np.float) >= 0.0]
            df = df[df['speed_nn'].values.astype(np.float) < 1.34]
            p0 = df.groupby('FR')['speed_nn'].max()
            p0 = p0[p0.values < 0.1].values.shape[0]
            # print(p0)
            if p0 > 0:
                p0count += 1

            # print(p0)
            # df = df[df['FR'] > fps * 10]
            df = df[df['X'] * df['X'] + df['Y'] * df['Y'] < 0.5 ** 2]
            v_nn_mean = np.append(v_nn_mean, df['speed_nn'].values.astype(np.float))
            # df = df[abs(df['X']) < 0.5]
            # df = df[df['Y'] > 1]

            # df = df[df['Y'] < 1]

            angle_nn = np.append(angle_nn, np.arccos(df['ANGLE_int_nn'].values.astype(np.float)) * 180. / 3.1415)
        print(p0count)
        # plt.hist(v_nn_mean,bins=50)
        # plt.show()
        df_anal = pd.DataFrame({"angle": angle_nn, "speed": v_nn_mean})

        dat_name = path + resultfolder + "dist_" + af.b_data_name(2 * lin_var[test_var][bi], 3) + ".csv"
        df_anal.to_csv(dat_name)
        x, bins, p = plt.hist(angle_nn,bins = 18, density=True)
        plt.savefig(path + 'plots/angledist/angle_' + af.b_data_name(2 * lin_var[test_var][bi], 3) + '.png')
        plt.clf()
        x, bins, p = plt.hist(v_nn_mean,bins = 18, density=True)
        plt.savefig(path + 'plots/speeddist/speed_' + af.b_data_name(2 * lin_var[test_var][bi], 3) + '.png')
        plt.clf()
        #nphist = np.histogram(v_nn_mean, density=True)
        # plt.plot(nphist[1][:-1],np.log(-np.log(1-nphist[0])), marker = "o", linestyle='none')
        #plt.plot(nphist[1][:-1], nphist[0], marker="o", linestyle='none')
        # plt.xscale("log")
        # plt.yscale("log")
        # print(v_nn_mean)
        v_0_list.append(v_nn_mean.mean())
        v_0_mean[bi] = v_nn_mean.mean()
        v_0_var[bi] = v_nn_mean.std()
        print(v_nn_mean.mean())
        angle_nn_mean[bi] = angle_nn.mean()
        angle_nn_var[bi] = angle_nn.std()

        print(angle_nn.mean())
        bi += 1
