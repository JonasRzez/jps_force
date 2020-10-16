import pandas as pd
import AnalFunctions as af
import numpy as np
import matplotlib.pyplot as plt
import os
#matplotlib.rcParams['text.usetex'] = True
import math as m
#import matplotlib


"""params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': True,
    'figure.figsize': [5, 5],
    'font.family': 'serif',
}
matplotlib.rcParams.update(params)"""

path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()
af.file_writer(path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var)

file_path = "../../expdata/"
sl = "/"

def add_wall(line_width,bi):
    wall1 = plt.Line2D((-bi / 2, -bi / 2), (0, 100), lw=line_width)
    wall2 = plt.Line2D((bi / 2, bi / 2), (0, 100), lw=line_width)
    wall3 = plt.Line2D((-bi / 2, -0.45), (0, 0), lw=line_width)
    wall4 = plt.Line2D((0.45, bi / 2), (0, 0), lw=line_width)
    wall5 = plt.Line2D((0.45, 0.25), (0., -0.15), lw=line_width)
    wall6 = plt.Line2D((-0.45, -0.25), (0., -0.15), lw=line_width)
    wall7 = plt.Line2D((0.25, 0.25), (-0.15, -1.0), lw=line_width)
    wall8 = plt.Line2D((-0.25, -0.25), (-0.15, -1.), lw=line_width)
    plt.gca().add_line(wall1)
    plt.gca().add_line(wall2)
    plt.gca().add_line(wall3)
    plt.gca().add_line(wall4)
    plt.gca().add_line(wall5)
    plt.gca().add_line(wall6)
    plt.gca().add_line(wall7)
    plt.gca().add_line(wall8)
os.system("mkdir " + path + "plots")
os.system("mkdir " + path + "plots/heatmaps")
T_test_list = lin_var[test_var2]
lattice_type = 'jule'
runs_tested = N_runs
traj_testvar2 = []

col = ["FR", "X", "Y"]

for T_test in T_test_list:
    folder_frame_frac = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'])
    b_folder = np.array(folder_frame.loc[folder_frame[test_str2] == T_test]['b'])

    loc_list = [[path + folder + sl + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(runs_tested)] for folder, bi in zip(folder_frame_frac, b_folder)]
    print(np.array(loc_list).shape)
    print("load trajectories " + str(T_test))
    trajectory_frame = [[loc for loc in loc_list_runs if os.path.isfile(loc)] for loc_list_runs in loc_list]
    # print(trajectory_frame)
    traj_testvar2.append(trajectory_frame)
    print("/load trajectories")

print("start heat map loop")
fps = 16
min_t = 10
max_t = 30
fwhm = 0.17
a = fwhm * m.sqrt(2) / (2 * m.sqrt(2 * m.log(2)))

array_min = int(min_t * fps)
array_max = int(max_t * fps)
j_list = range(lin_var[test_var2].shape[0])
x_array = np.linspace(-2.5, 2.5, 70)
y_array = np.linspace(-1.0, 5., 45)
for traj_test_var,j in zip(traj_testvar2,j_list):
    folder = test_str2 + "_"+ af.b_data_name(lin_var[test_var2][j], 3)
    os.system("mkdir " + path + "plots/heatmaps/" + folder)
    test_var_count = 0

    for loc_run,bi in zip(traj_test_var,b_folder):

        dens_matrix_runs = []
        print("*****************<calc density>*****************")
        for loc in loc_run:
            if os.stat(loc).st_size == 0:
                print("WARNING: file" + loc + " is empty")
                continue
            traj_i = pd.read_csv(loc, sep="\s+", header=0, comment="#",usecols=col)
            dens_matrix_list = []
            for time_point in np.arange(array_min, array_max, fps):
                x_i = traj_i[traj_i['FR'] == time_point]['X'].values
                y_i = traj_i[traj_i['FR'] == time_point]['Y'].values
                D = af.normal_matrix(x_i, y_i, x_array, y_array, a)
                dens_matrix_list.append(D)

            dens_matrix_np = np.array(dens_matrix_list)
            dens_matrix = dens_matrix_np.mean(axis=0)
            dens_matrix_runs.append(dens_matrix)
        print(np.array(dens_matrix_runs).shape)
        print("*****************</calc density>*****************")

        dens_matrix_mean = np.array(dens_matrix_runs).mean(axis=0)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-0.5, 5])

        x, y = np.meshgrid(x_array, y_array)
        z = dens_matrix_mean[:-1, :-1]

        z_min, z_max = np.abs(z).min(), np.abs(z).max()
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_yticks(ticks = [0,2,4] , minor=False)
        ax.set_xticks(ticks = [-2,0,2] , minor=False)
        ax.tick_params(axis='both', which='major', labelsize=30)

        c = ax.pcolormesh(x, y, z, cmap='hot', vmin=z_min, vmax=12)
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        cbar = fig.colorbar(c, ax=ax,ticks = [0,4,8,12])

        cbar.ax.tick_params(labelsize=30)

        print(z.mean())
        print("corridor width = ", bi)
        line_width = 2.5
        add_wall(line_width,bi*2)
        name_var = lin_var[test_var][test_var_count]
        test_var_count += 1
        plot_name = "heat_dens_" + af.b_data_name(name_var, 3) + ".pdf"

        plt.savefig(path + "plots/heatmaps/" + folder + "/" + plot_name)
        plt.close(fig=None)
