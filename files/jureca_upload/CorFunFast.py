import numpy as np
from matplotlib import pyplot as plt
import math as m

from scipy.integrate import simps
from multiprocessing import Pool
import pandas as pd
#import csv
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib as mpl

def latex_writer(file_name, caption, begin, end, overwrite):
    if overwrite:
        latex = open('../../expdata/plots/correlation_plots', 'w')
    else:
        latex = open('../../expdata/plots/correlation_plots', 'a')
    if begin:
        latex.write('\\begin{figure}[H]\n')
        latex.write('\\centering\n')
        latex.write('\\begin{minipage}{.5\\textwidth}\n')
        latex.write('\\centering\n')
        latex.write('\\includegraphics[width=1.0\\textwidth]{' + file_name + '.pdf}\n')
        latex.write('\\captionof{figure}{' + caption + '}\n')
        latex.write('\\end{minipage}%\n')
    if end:
        latex.write('\end{figure}\n')
    latex.close()


def smoothdirac(x, a):
    return 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2)


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
    #return (simps(simps(rho_matrix, x_array), y_array))


def two_point_correlation(lattice_x, lattice_y, x_array, y_array, r_array, phi_array, a):
    x_dens = np.array(
        [lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])

    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])

    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))

    rho_tensor2_x = np.array([[densty1d(x_dens - r * m.cos(phi), a) for phi in phi_array] for r in
                              r_array])  # calculating all points for density map
    rho_tensor2_y = np.array([[densty1d(y_dens - r * m.sin(phi), a) for phi in phi_array] for r in r_array])

    self_cor = np.array([[np.matmul((rho_matrix_x * rho_shift_matrix_x),
                                    np.transpose((rho_matrix_y * rho_shift_matrix_y))) for
                          rho_shift_matrix_x, rho_shift_matrix_y in zip(rho_tensor_x, rho_tensor_y)] for
                         rho_tensor_x, rho_tensor_y in zip(rho_tensor2_x, rho_tensor2_y)])

    rho_tensor2 = np.array([[np.matmul(rho_matrix_x, np.transpose(rho_matrix_y)) for
                             rho_matrix_x, rho_matrix_y in zip(rho_tensor_x, rho_tensor_y)] for
                            rho_tensor_x, rho_tensor_y in zip(rho_tensor2_x, rho_tensor2_y)])

    g_tensor2 = np.array(
        [[np.multiply(rho_matrix, rho_r_matrix) for rho_r_matrix in rho_r_tensor] for rho_r_tensor in rho_tensor2])

    print("Calculated two point correlation function")
    return [g_tensor2, rho_tensor2, self_cor]


def lattice_data(file_path,data_name, data_type,t_min,t_max,fps_n):
    data = pd.read_csv(file_path + data_name + '.txt', sep="\s+", header=0)
    data_frames = np.array(data["frame"])
    data_id = np.array(data["id"])

    max_frame = data_frames.max()
    max_id = data_id.max()
    print("N_ped = ", max_id)

    print(data.head())
    print("max_frame = ", max_frame)
    data_x_new = []
    data_y_new = []
    data_id_new = []
    data_frames_new = []
    for id in np.arange(1, max_id + 1):
        x_i = np.array(data[data['id'] == id]['x/m'])
        x_f = np.array(data[data['id'] == id]['frame'])
        y_i = np.array(data[data['id'] == id]['y/m'])

        if data_type == 'jule':
            if x_i.shape[0] < max_frame:
                diff = max_frame - x_i.shape[0]
                x_nan = [np.nan for i in np.arange(0, diff)]
                x_i = np.append(x_i, x_nan)
                y_i = np.append(y_i, x_nan)
                x_f = np.append(x_f, np.arange(x_f[-1] + 1, x_f[-1] + diff + 1))
                x_id = [id for i in np.arange(0, x_i.shape[0])]

            else:
                x_id = np.array(data[data['id'] == id][
                                    'id'])  # deletes the last frame of the person with maximal frames saved to unify the length of all frames
                x_id = x_id[0:-1]
                x_i = x_i[0:-1]
                x_f = x_f[0:-1]
                y_i = y_i[0:-1]
        else:
            diff = max_frame - x_i.shape[0]
            x_nan = [np.nan for i in np.arange(0, diff)]
            x_i = np.append(x_i, x_nan)
            y_i = np.append(y_i, x_nan)
            x_f = np.append(x_f, np.arange(x_f[-1] + 1, x_f[-1] + diff + 1))
            x_id = [id for i in np.arange(0, x_i.shape[0])]

        data_x_new = np.append(data_x_new, x_i)
        data_id_new = np.append(data_id_new, x_id)
        data_frames_new = np.append(data_frames_new, x_f)
        data_y_new = np.append(data_y_new, y_i)
    trajectory_dic = {'id': data_id_new, 'frame': data_frames_new, 'x': data_x_new, 'y': data_y_new}

    traj_frame = pd.DataFrame(trajectory_dic)

    fps = 25
    #fps_n = 10
    #t_min = 10
    #t_max = 16
    x_dic = {}
    y_dic = {}
    for i in np.arange(1, max_id):
        x_col = np.array(traj_frame[traj_frame['id'] == i]['x'])
        y_col = np.array(traj_frame[traj_frame['id'] == i]['y'])
        if data_name == 'bottleneck_sieben':
            x_col = x_col / 100
            y_col = y_col / 100
        x_dic[i] = x_col
        y_dic[i] = y_col
    traj_x_frame = pd.DataFrame(x_dic)
    traj_y_frame = pd.DataFrame(y_dic)
    #index_min = t_min * fps
    #index_max = t_max * fps

    mean_x_traj = np.array(traj_x_frame)[t_min * fps:t_max * fps]
    mean_y_traj = np.array(traj_y_frame)[t_min * fps:t_max * fps]

    mean_x_traj = mean_x_traj[::fps_n]
    mean_y_traj = mean_y_traj[::fps_n]

    #mean_x_traj = np.array([np.array(traj_x_frame[t:t + fps_n].mean()) for t in np.arange(index_min, index_max, fps_n)])
    lattice_x = np.array([[x for x in x_line if 1 - np.isnan(x)] for x_line in mean_x_traj])
    #mean_y_traj = np.array([np.array(traj_y_frame[t:t + fps_n].mean()) for t in np.arange(index_min, index_max, fps_n)])
    lattice_y = np.array([[y for y in y_line if 1 - np.isnan(y)] for y_line in mean_y_traj])
    return lattice_x, lattice_y


def correlation_data(lattice_x, lattice_y, x_array, y_array, r_array, phi_array, fwhm):
    a = fwhm * m.sqrt(2) / (2 * m.sqrt(2 * m.log(2)))

    normal_tensor = np.array(
        [normal(x_values, y_values, x_ar, y_ar, a) for x_values, y_values,x_ar,y_ar in zip(lattice_x, lattice_y,x_array,y_array)])
    # print('normal tensore = ', normal_tensor)
    print("Calculating <rho(r)rho(x-r)>")
    print("******************************<POOLING DATA>******************************")
    pool = Pool(processes=8)
    g_pool = np.array(
        [pool.apply_async(two_point_correlation, args=(x_values, y_values, x_ar, y_ar, r_array, phi_array, a)) for
         x_values, y_values,x_ar,y_ar in zip(lattice_x, lattice_y,x_array,y_array)])
    tensor_list = np.array([p.get() for p in g_pool])
    pool.close()
    g_tensor2_list = np.array([tensor[0] for tensor in tensor_list])
    rho_x_r_tensor_list = np.array([tensor[1] for tensor in tensor_list])

    print('rho_x_r_tensor.shape = ', rho_x_r_tensor_list.shape)

    self_cor_list = np.array([tensor[2] for tensor in tensor_list])
    print('self_core_lsit = ', self_cor_list.shape)
    #self_mean = self_cor_list.mean(axis)

    print("******************************<\POOLING DATA>******************************")
    #print('g_tensor2_list shape = ', g_tensor2_list.shape)
    rho_0 = normal_tensor.mean()
    print("Calculation mean of g_r_phi")

    g_r_phi = g_tensor2_list.mean(axis=0)
    #g_r_phi = np.array([[simps(simps(g_matrix, x_array[0]), y_array[0]) for g_matrix in g_matrix_phi] for g_matrix_phi in g_tensor2_list])
    print("Calculation mean of self_core")

    self_cor = self_cor_list.mean(axis=0).mean(axis=3).mean(axis=2)
    print("Calculation mean of rho_x_r_mean")

    rho_x_r_mean = rho_x_r_tensor_list.mean(axis=0).mean(axis=2).mean(axis=2)
    print('rho_x_r_mean.shape = ', rho_x_r_mean.shape)
    print("g_r_phi.shape ",g_r_phi.shape)
    g_r_phi_mean = g_r_phi.mean(axis=2).mean(axis=2)
    cor = g_r_phi_mean - self_cor
    norm = rho_0 * rho_x_r_mean
    print("calculation g_r")
    g_r = cor.mean(axis=1) / norm.mean(axis=1)
    #g_r_2 = cor.mean(axis=1)/rho_0**2
    plt.plot(r_array, g_r)
    #plt.plot(r_array, g_r_2)

    plt.show()

    return cor, norm


def plot_correlation(g_tensor, data_name, r_array, phi_array, para, ow_bool):
    # normal_mean = normal_tensor.mean()
    mean_phi_g_tensor = g_tensor.mean(axis=1)# mean_g_tensor.mean(axis=1)

    # mean_phi_g_tensor = g_tensor.mean(axis = 0)
    N_phi = int(round(sum(phi_array)))
    file_name = 'G(R)' + data_name + "_N_phi=_" + str(N_phi)
    file_name_log = 'G(R)_log_' + data_name + "_N_phi=_" + str(N_phi)
    caption = 'G(r) for N phi = ' + str(N_phi) + ' N ped = ' + para[0] + ' N x y = ' + para[1] + ' ' + para[2] + \
              ' Mot = ' + para[3] + 'x lim = ' + para[4] + ' ' + para[5] + 'y lim = ' + para[5] + ' ' + para[6]
    plt.plot(r_array, mean_phi_g_tensor)  # / normal_mean)

    plt.xlabel("r")
    plt.ylabel("G(r)")
    plt.title("Two Point Correlation Function")
    plt.savefig('../../expdata/plots/' + file_name + '.pdf')
    latex_writer(file_name, caption, begin=True, end=False, overwrite=ow_bool)

    plt.show()
    plt.plot(r_array, mean_phi_g_tensor)

    plt.yscale('log')
    plt.savefig('../../expdata/plots/' + file_name_log + '.pdf')

    plt.show()
    latex_writer(file_name_log, caption, begin=False, end=True, overwrite=False)

    x_heat = np.array([[r * m.cos(phi) for phi in phi_array] for r in r_array])
    y_heat = np.array([[r * m.sin(phi) for phi in phi_array] for r in r_array])

    g_matrix_phi1 = g_tensor[:-1, :-1]
    print(g_matrix_phi1.min())
    print(g_matrix_phi1.max())

    z_min, z_max = g_matrix_phi1.min(), g_matrix_phi1.max()
    print(x_heat.shape, y_heat.shape, g_matrix_phi1.shape)
    print(z_min, z_max)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x_heat, y_heat, g_matrix_phi1, cmap='Blues', vmin=z_min, vmax=z_max)
    ax.set_title('two point correlation')
    print(z_min, z_max)
    # set the limits of the plot to the limits of the data
    ax.axis([x_heat.min(), x_heat.max(), y_heat.min(), y_heat.max()])
    print(x_heat.min())
    fig.colorbar(c, ax=ax)
    plt.savefig('../../expdata/plots/G_Heat_ ' + data_name + "_N_phi=_" + str(r_array.shape[0]) + '.pdf')

def map_maker(lattice_x,lattice_y):
    x_max = 0
    x_min = 1000
    y_max = 0
    y_min = 1000
    for x_list,y_list in zip(lattice_x, lattice_y):
        x_max_i = np.array(x_list).max()
        x_min_i = np.array(x_list).min()
        y_max_i = np.array(y_list).max()
        y_min_i = np.array(y_list).min()
        if x_max_i > x_max:
            x_max = x_max_i
        if x_min_i < x_min:
            x_min = x_min_i
        if y_max_i > y_max:
            y_max = y_max_i
        if y_min_i < y_min:
            y_min = y_min_i

    x_array = np.linspace(x_min,x_max,1000)
    y_array = np.linspace(y_min,y_max,1000)
    '''
    r_max_p = m.sqrt(x_max**2 + y_max ** 2)
    r_max_n = m.sqrt(x_min**2 + y_min ** 2)
    if r_max_p > r_max_n:
        r_max = r_max_p
    else:
        r_max = r_max_n
    #r_array = (0,r_max,1000)
    #phi_array = (0,2*m-pi,600)
    '''
    return x_array, y_array

def density_map(lattice_x,lattice_y,x_array,y_array,fwhm):
    a = fwhm * m.sqrt(2) / (2 * m.sqrt(2 * m.log(2)))
    rho_matrix_list = []
    for l_x,l_y in zip(lattice_x,lattice_y):
        x_dens = np.array([l_x - x for x in x_array])
        y_dens = np.array([l_y - y for y in y_array])
        rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
        rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
        rho_matrix = np.matmul(rho_matrix_x,np.transpose(rho_matrix_y))
        rho_matrix_list.append(rho_matrix)
    rho_tensor = np.array(rho_matrix_list)
    print(rho_tensor.shape)
    return rho_tensor

def heat_map(heat_matrix,x_heat,y_heat):
    heat_matrix_1 = heat_matrix[:-1, :-1]
    print(heat_matrix_1.min())
    print(heat_matrix_1.max())

    z_min, z_max = heat_matrix_1.min(), heat_matrix_1.max()
    print(x_heat.shape, y_heat.shape, heat_matrix_1.shape)
    print(z_min, z_max)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x_heat, y_heat, heat_matrix_1, cmap='Blues', vmin=z_min, vmax=z_max)
    ax.set_title('two point correlation')
    print(z_min, z_max)
    # set the limits of the plot to the limits of the data
    ax.axis([x_heat.min(), x_heat.max(), y_heat.min(), y_heat.max()])
    print(x_heat.min())
    fig.colorbar(c, ax=ax)
    plt.show()
    #plt.savefig('../../expdata/plots/G_Heat_ ' + data_name + "_N_phi=_" + str(r_array.shape[0]) + '.pdf')

def lattice_animation(data_name, data_type):
    data = pd.read_csv('../../expdata/' + data_name + '.txt', sep="\s+", header=0)
    data_frames = np.array(data["frame"])
    data_id = np.array(data["id"])

    max_frame = data_frames.max()
    max_id = data_id.max()
    print(data.head())
    print("max_frame = ", max_frame)
    data_x_new = []
    data_y_new = []
    data_id_new = []
    data_frames_new = []
    for id in np.arange(1, max_id + 1):
        x_i = np.array(data[data['id'] == id]['x/m'])
        x_f = np.array(data[data['id'] == id]['frame'])
        y_i = np.array(data[data['id'] == id]['y/m'])

        if data_type == 'jule':
            if x_i.shape[0] < max_frame:
                diff = max_frame - x_i.shape[0]
                x_nan = [np.nan for i in np.arange(0, diff)]
                x_i = np.append(x_i, x_nan)
                y_i = np.append(y_i, x_nan)
                x_f = np.append(x_f, np.arange(x_f[-1] + 1, x_f[-1] + diff + 1))
                x_id = [id for i in np.arange(0, x_i.shape[0])]

            else:
                x_id = np.array(data[data['id'] == id][
                                    'id'])  # deletes the last frame of the person with maximal frames saved to unify the length of all frames
                x_id = x_id[0:-1]
                x_i = x_i[0:-1]
                x_f = x_f[0:-1]
                y_i = y_i[0:-1]
        else:
            diff = max_frame - x_i.shape[0]
            x_nan = [np.nan for i in np.arange(0, diff)]
            x_i = np.append(x_i, x_nan)
            y_i = np.append(y_i, x_nan)
            x_f = np.append(x_f, np.arange(x_f[-1] + 1, x_f[-1] + diff + 1))
            x_id = [id for i in np.arange(0, x_i.shape[0])]

        data_x_new = np.append(data_x_new, x_i)
        data_id_new = np.append(data_id_new, x_id)
        data_frames_new = np.append(data_frames_new, x_f)
        data_y_new = np.append(data_y_new, y_i)
    trajectory_dic = {'id': data_id_new, 'frame': data_frames_new, 'x': data_x_new, 'y': data_y_new}

    traj_frame = pd.DataFrame(trajectory_dic)

    fps = 25
    fps_n = 10
    t_min = 5
    t_max = 14
    x_dic = {}
    y_dic = {}
    print("N_ped = ", max_id)
    for i in np.arange(1, max_id):
        x_col = np.array(traj_frame[traj_frame['id'] == i]['x'])
        y_col = np.array(traj_frame[traj_frame['id'] == i]['y'])
        if data_name == 'bottleneck_sieben':
            x_col = x_col / 100
            y_col = y_col / 100
        x_dic[i] = x_col
        y_dic[i] = y_col
    lattice_x = pd.DataFrame(x_dic)
    lattice_y = pd.DataFrame(y_dic)
    return lattice_x, lattice_y


def animation(data_name, data_type):

    # x_data_raw,y_data_raw = lattice_animation(data_name_new[2], 'navarra')
    x_data_raw, y_data_raw = lattice_animation(data_name, data_type)

    x_data_raw = np.array(x_data_raw)
    y_data_raw = np.array(y_data_raw)
    for i in np.arange(0, x_data_raw.shape[0]):
        for j in np.arange(0, x_data_raw.shape[1]):
            if m.isnan(x_data_raw[i, j]):
                x_data_raw[i, j] = -1000
                y_data_raw[i, j] = -1000

    x_data = np.array([x_data_raw[i, :] for i in np.arange(0, x_data_raw.shape[0], 1)])
    y_data = np.array([y_data_raw[i, :] for i in np.arange(0, y_data_raw.shape[0], 1)])
    print(x_data.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 5)
    scatter = ax.scatter(x_data[0], y_data[0], zorder=5)  # Scatter plot

    players = []
    for n in range(len(x_data[0])):
        # as there are always 3 circles, append all three patches as a list at once
        players.append([mpl.patches.Circle((x_data[0, n], y_data[0, n]), radius=0.1, color='black', lw=1, alpha=0.8, zorder=4), ])

    ##adding patches to axes
    for player in players:
        for circle in player:
            ax.add_patch(circle)


    def animate(i):
        scatter.set_offsets(np.c_[x_data[i, :], y_data[i, :]])

        ##updating players:
        for n, player in enumerate(players):
            for circle in player:
                circle.center = (x_data[i, n], y_data[i, n])


    ani = animation.FuncAnimation(fig, animate, frames=len(x_data),
                                  interval=25, blit=False)

    HTML(ani.to_html5_video())
