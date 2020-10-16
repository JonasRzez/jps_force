import numpy as np
from matplotlib import pyplot as plt
import math as m
from scipy.integrate import simps
from multiprocessing import Pool
import pandas as pd
import sys

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

def lattice_data(data_name,data_type):
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
                x_id = np.array(data[data['id'] == id]['id'])  # deletes the last frame of the person with maximal frames saved to unify the length of all frames
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
    fps_n = 15
    t_min = 5
    t_max = 14
    x_dic = {}
    y_dic = {}
    print("N_ped = ", max_id)
    for i in np.arange(1, max_id):
        x_col = np.array(traj_frame[traj_frame['id'] == i]['x'])
        y_col = np.array(traj_frame[traj_frame['id'] == i]['y'])
        if data_name == 'Bottleneck_sieben':
            x_col = x_col / 100
            y_col = y_col / 100
        x_dic[i] = x_col
        y_dic[i] = y_col
    traj_x_frame = pd.DataFrame(x_dic)
    traj_y_frame = pd.DataFrame(y_dic)
    index_min = t_min * fps
    index_max = t_max * fps

    mean_x_traj = np.array([np.array(traj_x_frame[t:t + fps_n].mean()) for t in np.arange(index_min, index_max, fps_n)])
    lattice_x = np.array([[x for x in x_line if 1 - np.isnan(x)] for x_line in mean_x_traj])
    mean_y_traj = np.array([np.array(traj_y_frame[t:t + fps_n].mean()) for t in np.arange(index_min, index_max, fps_n)])
    lattice_y = np.array([[y for y in y_line if 1 - np.isnan(y)] for y_line in mean_y_traj])
    return lattice_x, lattice_y


def smoothdirac(x):
    a = 0.1
    return 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2)


def densty1d(delta_x, a):
    return np.array(list(map(lambda x: 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2), delta_x)))


def normal(l_x, l_y, x, y, a):
    delta_x = l_x - x  # calculate the distant of lattice pedestrians to the measuring lattice
    delta_y = l_y - y
    rho_array_x = np.array(densty1d(delta_x, a))  # density matrix is calculated
    rho_array_y = np.array(densty1d(delta_y, a))  # density matrix is calculated
    rho = np.dot(rho_array_x, rho_array_y)
    return rho


def two_point_correlation(l_x, l_y, x, y, r_array, phi_array, a):
    delta_x = l_x - x  # calculate the distant of lattice pedestrians to the measuring lattice
    delta_y = l_y - y

    rho_array_x = densty1d(delta_x, a)   # density matrix is calculated
    rho_array_y = densty1d(delta_y, a)

    rho_0 = np.dot(rho_array_x, rho_array_y)

    rho_tensor_x = np.array([[densty1d(delta_x - r * m.cos(phi),a) for phi in phi_array] for r in r_array])  # calculating all points for density map
    rho_tensor_y = np.array([[densty1d(delta_y - r * m.sin(phi),a) for phi in phi_array] for r in r_array])

    ### new test for matrix method ############

    g_matrix = np.array([[rho_0 * np.dot(rho_array_x, rho_array_y) for
                                   rho_array_x, rho_array_y in zip(rho_matrix_x, rho_matrix_y)] for
                                  rho_matrix_x, rho_matrix_y in zip(rho_tensor_x, rho_tensor_y)])
    '''
    rho_shift_tensor2 = np.array([[np.matmul(rho_matrix_x, np.transpose(rho_matrix_y)) for
                                   rho_matrix_x, rho_matrix_y in zip(rho_tensor_x, rho_tensor_y)] for
                                  rho_tensor_x, rho_tensor_y in zip(rho_tensor2_x, rho_tensor2_y)]) # method where rho_matrix isn't multiplied directly to calculate g(r)'''

    print("Calculated two point correlation function")
    return g_matrix


def correlation_data(lattice_x,lattice_y,x,y,r_array,phi_array,fwhm):
    a = fwhm * m.sqrt(2) / (2 * m.sqrt(2 * m.log(2)))
    normal_tensor = np.array([normal(x_values, y_values, x, y, a) for x_values, y_values in zip(lattice_x, lattice_y)])

    #print('normal tensore = ', normal_tensor)
    print("Calculating <rho(x)rho(x-r)>")
    print("pooling data")
    pool = Pool(processes=16)
    g_pool = np.array(
        [pool.apply_async(two_point_correlation, args=(x_values, y_values, x, y, r_array, phi_array, a)) for
         x_values, y_values in zip(lattice_x, lattice_y)])
    g_matrix_list = np.array([p.get() for p in g_pool])
    g_matrix = g_matrix_list.mean(axis = 0)
    print('g_matrix = ' , g_matrix.shape, g_matrix_list.shape)
    #sys.exit("End of working programm")

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!POOLING DATA FINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    g_r_phi_mean = g_matrix.mean(axis=1)
    print('g_r_phi_shape = ',g_r_phi_mean.shape)
    rho_list = np.array([l_x.shape[0] / (l_x.max() - l_x.min()) * (l_y.max() - l_y.min()) for l_x, l_y in zip(lattice_x, lattice_y)])
    rho_0 = rho_list.mean()

    delta_x_r = densty1d(r_array, a)
    rho_delta = np.array([delta_x_r for phi in phi_array])
    plt.plot(r_array,rho_delta[0])
    plt.show()
    rho_delta = rho_delta[0]
    g_r_phi = g_r_phi_mean/rho_0**2 - rho_delta/rho_0

    return g_r_phi


def plot_correlation(g_r,data_name,r_array,phi_array,para,ow_bool):

    #mean_phi_g_tensor = g_tensor.mean(axis = 1) #mean_g_tensor.mean(axis=1)
    N_phi = int(round(sum(phi_array)))
    file_name = 'G(R)' + data_name + "_N_phi=_" + str(N_phi)
    file_name_log = 'G(R)_log_' + data_name + "_N_phi=_" + str(N_phi)
    caption = 'G(r) for N phi = ' + str(N_phi) + ' N ped = ' + para[0] + ' N x y = ' + para[1] + ' ' + para[2] + \
              ' Mot = ' + para[3] + 'x lim = ' + para[4] + ' ' + para[5] + 'y lim = ' + para[5] + ' ' + para[6]
    plt.plot(r_array, g_r )

    plt.xlabel("r")
    plt.ylabel("G(r)")
    plt.title("Two Point Correlation Function")
    plt.savefig('../../expdata/plots/' + file_name + '.pdf')
    latex_writer(file_name, caption, begin = True , end = False, overwrite = ow_bool)

    plt.show()
    plt.plot(r_array, g_r)

    plt.yscale('log')
    plt.savefig('../../expdata/plots/' + file_name_log + '.pdf')

    plt.show()
    latex_writer(file_name_log, caption, begin = False , end = True, overwrite = False)

   # x_heat = np.array([[r * m.cos(phi) for phi in phi_array] for r in r_array])
   # y_heat = np.array([[r * m.sin(phi) for phi in phi_array] for r in r_array])

"""
    g_matrix_phi1 = mean_g_tensor[:-1, :-1] / normal_mean
    print(g_matrix_phi1.min())
    print(g_matrix_phi1.max())

    # In[ ]:

    z_min, z_max = g_matrix_phi1.min(), g_matrix_phi1.max()
    print(x_heat.shape, y_heat.shape, g_matrix_phi1.shape)
    print(z_min, z_max)

    # In[ ]:

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x_heat, y_heat, g_matrix_phi1, cmap='Blues', vmin=z_min, vmax=z_max)
    ax.set_title('two point correlation')
    print(z_min, z_max)
    # set the limits of the plot to the limits of the data
    ax.axis([x_heat.min(), x_heat.max(), y_heat.min(), y_heat.max()])
    print(x_heat.min())
    fig.colorbar(c, ax=ax)
    plt.savefig('../../expdata/plots/G_Heat_ ' + data_name + "_N_phi=_" + str(N_angle) + '.pdf')
"""
