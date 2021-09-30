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
import cmath
from itertools import groupby

def trajReduce(second,df,fps,dummy):
    df_t = df[df['FR'] == int(second * fps)]
    x = df_t['X'].to_numpy()#/100
    y = df_t['Y'].to_numpy()#/100
    speed_nn = df_t['speed_nn'].to_numpy()
    ids = df_t['ID'].to_numpy()
    print(ids.shape)
    if dummy:
        y_add = np.array([-0.1 for i in range(100)])
        x_add = np.linspace(-25,25,100)
        speed_add = np.array([0 for i in range(100)])
        ids_add = np.array([max(ids) + 1000 for i in range(100)])
        x = np.append(x,x_add)
        y = np.append(y,y_add)
        speed_nn = np.append(speed_nn,speed_add)
        ids = np.append(ids,ids_add)
    print(np.array(ids).shape,np.array(x).shape,np.array(y).shape,np.array(speed_nn).shape)
    XY = pd.DataFrame({'id':ids,'x':x,'y':y,'speed_nn':speed_nn})
    return XY

def orderfetch(second,ni,df,fps,test_str2,T_test,csvname,filtered,box):
    XY = trajReduce(second,df,fps,True)
    XYLocalAppend(XY,second,test_str2,T_test,filtered,box)
    length = XY['x'].shape[0]
    XY['i'] = listMaker(ni,length)
    XY.to_csv(csvname)
    
def filechecker(loc):
    continue_ = False
    if os.path.isfile(loc) == False:
        print("WARNING: file " + loc + " not found.")
        continue_ = True
    elif os.stat(loc).st_size == 0:
        print("WARNING: file " + loc + " is empty.")
        continue_ = True
    print(continue_)
    return continue_

def fileload(loc,load,col):
    if load:
        df = pd.read_csv(loc, sep="\s+", header=0, comment="#",skipinitialspace=True, usecols=col)
    else:
        df = "none"
    return df

def localBondOrientationFactorneu(p_index,tri,filtered,filterbox):
    neigh = find_neighbors(p_index,tri,filtered,filterbox)
    Nb = neigh.shape[0]
    bond_orientation_list = np.empty(Nb)
    exp_sum = 0
    if neigh.shape[0] > 0:
        for n in range(Nb - 1):
            bond1 = tri.points[neigh[n]] - tri.points[p_index]
            bond2 = tri.points[neigh[n+1]] - tri.points[p_index]
            leng1 = np.linalg.norm(bond1)
            leng2 = np.linalg.norm(bond2)
            bond1 = bond1/leng1
            bond2 = bond2/leng2
            #print(round(np.dot(bond1,bond2),4))
            #print(np.dot(bond1,bond2) - pi/6)
            #bond_angle = np.cos(6 * np.arccos(round(np.dot(bond1,bond2),4)))
            #cos_sum += bond_angle
            angle =  np.arccos(round(np.dot(bond1,bond2),4))
            bond_angle = cmath.exp(6j * angle)
            exp_sum += bond_angle
            nb_sum = 1/Nb * exp_sum

        bond1 = tri.points[neigh[0]] - tri.points[p_index]
        bond2 = tri.points[neigh[-1]] - tri.points[p_index]
        leng1 = np.linalg.norm(bond1)
        leng2 = np.linalg.norm(bond2)
        bond1 = bond1/leng1
        bond2 = bond2/leng2
        #print(round(np.dot(bond1,bond2),4))
        #print(np.dot(bond1,bond2) - pi/6)
        angle =  np.arccos(round(np.dot(bond1,bond2),4))
        bond_angle = cmath.exp(6j * angle)
        exp_sum += bond_angle
        nb_sum = 1/Nb * exp_sum
        #print(nb_sum)
        #bond_angle = np.cos(6 * np.arccos(round(np.dot(bond1,bond2),4)))
        #cos_sum += bond_angle
        #nb_sum = 1/Nb * cos_sum
        nb_sum = np.sqrt(nb_sum.real**2 + nb_sum.imag**2)
    else:
        nb_sum = 0
    return nb_sum

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


def foldermaker(foldername):
    folder = ''
    folderold = ''
    for l in foldername:
        if l != "/":
            folder += l
        else:
            print(folder)
            if folderold == '':
                os.system("mkdir " + path + "plots/structure/" + folder)
            else:
                os.system("mkdir " + path + "plots/structure/"+ folderold + "/" + folder)
            folderold = folder
            folder = ''


def OrderFieldPlot(order_matrix_2,x_heat,y_heat,XY_red,foldername,datname,form,value_max,value_min,colorscheme,savenpy):
    #print(np.array(order_matrix_2).mean())
    #print(XY_red['Bf'].values.mean())
    if foldername != False:
        foldermaker(foldername)
    #os.system("mkdir " + path + "plots/structure" + foldername)
    if savenpy:
        np.save(path + "plots/structure/"+ foldername + datname + ".npy",np.array(order_matrix_2))
    ordermatrix = np.array(order_matrix_2).mean(axis=1).mean(axis=0)[:-1, :-1]
    z_min, z_max = ordermatrix.min(), ordermatrix.max()

    fig, ax = plt.subplots(figsize = (13, 10))
    if value_max == "none":
        value_max = z_max
    if value_min == "none":
        value_min = z_min
    c = ax.pcolormesh(x_heat, y_heat, ordermatrix, cmap=colorscheme,vmin = value_min, vmax = value_max)
    # set the limits of the plot to the limits of the data
    ax.axis([x_heat.min(), x_heat.max(), y_heat.min(), y_heat.max()])
    fig.colorbar(c, ax=ax)
    if foldername != False:
        plt.savefig(path + "plots/structure/"+ foldername + datname + "." + form)
    plt.show()
    
def MeanBondFacotrR(order_matrix_2,x_heat,y_heat,foldername,datname,form,ylimit):
    foldermaker(foldername)
    xymesh = np.meshgrid(x_heat,y_heat)
    x_r = np.round(xymesh[0].flatten(),1)
    y_r = np.round(xymesh[1].flatten(),1)
    orderlist = np.array(order_matrix_2).mean(axis=1).mean(axis=0).flatten()
    r = np.round(np.sqrt(x_r ** 2 + y_r ** 2),0)
    orderframe = pd.DataFrame({'r':r, 'order':orderlist})
    new_r = np.arange(0,10,1)
    ordermean = []
    for r in new_r:
        ordermean.append(orderframe[orderframe['r'] == r]["order"].values.mean())
    return new_r, ordermean

def find_neighbors(pindex, tri,filtered,filterbox):
    neigh = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex+1]]
    if filtered:
        return neighboursFilter(neigh,filterbox,tri)
    else:
        return neigh

def indexDistance(p_index,tri,filtered,filterbox):
    neigh = find_neighbors(p_index,tri,filtered,filterbox)
    dist_list = np.empty(neigh.shape[0])
    for n in range(neigh.shape[0]):
        bond = tri.points[neigh[n]] - tri.points[p_index]
        dist = np.linalg.norm(bond)
        dist_list[n] = dist
    return dist_list
    
def localBondOrientationFactor(p_index,tri,filtered,filterbox):
    neigh = find_neighbors(p_index,tri,filtered,filterbox)
    Nb = neigh.shape[0]
    bond_orientation_list = np.empty(Nb)
    exp_sum = 0
    for n in range(Nb - 1):
        bond1 = tri.points[neigh[n]] - tri.points[p_index]
        bond2 = tri.points[neigh[n+1]] - tri.points[p_index]
        leng1 = np.linalg.norm(bond1)
        leng2 = np.linalg.norm(bond2)
        bond1 = bond1/leng1
        bond2 = bond2/leng2
        #print(round(np.dot(bond1,bond2),4))
        #print(np.dot(bond1,bond2) - pi/6)
        #bond_angle = np.cos(6 * np.arccos(round(np.dot(bond1,bond2),4)))
        angle =  np.arccos(np.dot(np.array([0,1]),np.array([1,0])))
        bond_angle = cmath.exp(6j * angle)
        
        exp_sum += bond_angle
    nb_sum = 1/(Nb -1) * exp_sum

    return nb_sum

def bondOrientationFactor(tri,N,filtered,filterbox):
    nb_sum = 0
    for p_index in range(N):
            #print(tri.points[p_index])
        neigh = find_neighbors(p_index,tri,filtered,filterbox)
        Nb = len(neigh)
        cos_sum = 0
        for n in range(Nb-1):
            bond1 = tri.points[neigh[n]] - tri.points[p_index]
            bond2 = tri.points[neigh[n+1]] - tri.points[p_index]
            leng1 = np.dot(bond1,bond1)
            leng2 = np.dot(bond2,bond2)
            bond1 = bond1/np.sqrt(leng1)
            bond2 = bond2/np.sqrt(leng2)
            #print(round(np.dot(bond1,bond2),4))
            bond_angle = np.cos(6 * np.arccos(round(np.dot(bond1,bond2),4)))
            #bond_angle = round(np.dot(bond1,bond2),4)

            #print(bond_angle)
            cos_sum += bond_angle
        nb_sum += 1/(Nb -1) * cos_sum

    return 1/N * nb_sum

def hexagonalLattice(N_in,a,mu,lb,hb,N):
    # (N_in/a)**2 is the number of points
    #a is point distance
    #mu is noise level
    #lb and hb are the bounds of the uniform distribution
    #N is the number of hexagonal lattices that are produced and written into a N X N_in^2/a^2 dimensional array
    N_particles = int(N_in * N_in/a**2)
    l_x = np.empty([N,N_particles])
    l_y = np.empty([N,N_particles])
    print(l_x.shape)
    print(l_y.shape)
    for i in range(N):
        h_list = np.arange(0,N_in,a)
        l_list = np.arange(0,N_in,a)
        x = np.array([(h/2 - 1/2 * l) for h in h_list for l in l_list])
        y = np.array([(m.sqrt(3)/2 * h + m.sqrt(3)/2 * l) for h in h_list for l in l_list])
        x_noise = mu * np.random.uniform(lb,hb ,x.shape[0])
        y_noise = mu * np.random.uniform(lb,hb,y.shape[0])
        x = x + x_noise
        y = y + y_noise

        l_x[i] = x
        l_y[i] = y
    
    return l_x,l_y

def latticeDistance(tri,box):
    dist = []
    dist_tri = []
    #check for box for distance
    for t in points[tri.simplices]:
        #print(t[0])
        boxcheck0 = boxchecker(t[0],box)
        boxcheck1 = boxchecker(t[1],box)
        boxcheck2 = boxchecker(t[2],box)
        if boxcheck0 + boxcheck1 + boxcheck2 == False:
            #print("outside the box")
            continue
            
        dist_i = []
        """print("distance")
        print(t[0],t[1])
        print(t[0]-t[1])"""
        if boxcheck0 and boxcheck1:
            d1 = t[0] - t[1]
            d1 = np.linalg.norm(d1)
            dist.append(d1)
            dist_i.append(d1)
        if boxcheck0 and boxcheck2:
            d2 = t[0] - t[2]
            d2 = np.linalg.norm(d2)
            dist.append(d2)
            dist_i.append(d2)
        if boxcheck1 and boxcheck2:
            d3 = t[1] - t[2]
            d3 = np.linalg.norm(d3)
            dist.append(d3)
            dist_i.append(d3)
        dist_tri.append(dist_i)
    return np.array(dist_tri), np.array(dist)

def delHexMeasure(dist_tri,dist):
    dist_mean = dist.mean()
    del_m = dist_tri - dist_mean
    del_m = np.abs(del_m)
    del_m = np.sum(del_m,axis = 1)
    del_m = del_m.mean()
    return del_m

def normal(lattice_x, lattice_y, x_array, y_array, a):
    x_dens = np.array(
        [lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])
    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.T

def densty1d(delta_x, a):
    return np.array(list(map(lambda x: 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2), delta_x)))

def densty1dWeight(delta_x, weight,a):
    return np.array(list(map(lambda var: np.sqrt(abs(var[1])) / (m.sqrt(m.pi) * a) * m.e ** (-var[0] ** 2 / a ** 2), zip(delta_x,weight))))

def densty1dWeightSign(delta_x, weight,a):
    return np.array(list(map(lambda var: np.sqrt(var[1]) / (m.sqrt(m.pi) * a) * m.e ** (-var[0] ** 2 / a ** 2) if var[1] > 0 else -np.sqrt(abs(var[1])) / (m.sqrt(m.pi) * a) * m.e ** (-var[0] ** 2 / a ** 2), zip(delta_x,weight) )))

def orderField(lattice_x,lattice_y,order,x_array,y_array,a):
    x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])
    rho_matrix_x = np.array([densty1dWeightSign(delta_x,order, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1dWeight(delta_y,order, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    order_matrix = rho_matrix.T/normal(lattice_x, lattice_y, x_array, y_array, a)
    
    return order_matrix
    
def localOrientationMeasures(tri,N,filtered,filterbox):
    neighbour_list = np.empty(N)
    neigh_dist_list = np.empty(N)
    local_bond_list = np.empty(N)
    mean_neighdist_list = np.empty(N)
    #dist_tri, dist = latticeDistance(tri,filterbox)
    #distmean = dist.mean()
    for p_index in range(N):
        neigh_dist = indexDistance(p_index,tri,filtered,filterbox)
        neighbour_list[p_index] = len(find_neighbors(p_index,tri,filtered,filterbox))
        neigh_dist_list[p_index] = np.var(neigh_dist/neigh_dist.mean())#np.sum(np.abs(np.round(neigh_dist/distmean - 1,3)))/neigh_dist.shape[0] #np.var(neigh_dist/neigh_dist.mean())#
        local_bond_list[p_index] = localBondOrientationFactorneu(p_index,tri,filtered,filterbox)
        mean_neighdist_list[p_index] = neigh_dist.mean()
    return neighbour_list, neigh_dist_list, local_bond_list, mean_neighdist_list
    
def pedReducer(XY,x_min,x_max,y_min,y_max,Nn_max):
    #XY_red = XY[XY['Nn'] < Nn_max]
    XY_red = XY[XY['x'] < x_max]
    XY_red = XY_red[XY_red['x'] > x_min]
    XY_red = XY_red[XY_red['y'] < y_max]
    XY_red = XY_red[XY_red['y'] > y_min]
    return XY_red

def filtration(var,filvar,fillist,i):
    if var < filvar:
        fillist.append(i)
    
def neighboursFilter(neighbours,filterbox,tri):
    #filgerbox[x_min,x_max,y_min,y_max]
    del_list = []

    for i in np.arange(neighbours.shape[0]):
        if boxchecker(tri.points[neighbours[i]],filterbox):
            del_list.append(i)
        
    neighbours = np.delete(neighbours,del_list)
    return neighbours

"""def trajReduce(second,df,fps,dummy):
    df_t = df[df['FR'] == second * fps]
    x = df_t['X'].values#/100
    y = df_t['Y'].values#/100
    speed_nn = df_t['speed_nn'].values
    ids = df_t['ID'].values
    if dummy:
        y_add = np.array([-0.1 for i in range(100)])
        x_add = np.linspace(-25,25,100)
        speed_add = np.array([0 for i in range(100)])
        ids_add = listMaker(ids.max() + 100)
        x = np.append(x,x_add)
        y = np.append(y,y_add)
        speed_nn = np.append(speed_nn,speed_add)
    print(np.array(ids).shape,np.array(x).shape,np.array(y).shape,np.array(speed_nn).shape)
    XY = pd.DataFrame({'id':ids,'x':x,'y':y,'speed_nn':speed_nn})
    return XY"""

def boxchecker(point,box):
    if point[0] < box[0]:
        return False
    elif point[0] > box[1]:
        return False
    elif point[1] < box[2]:
        return False
    elif point[1] > box[3]:
        return False
    else:
        return True
    
def dummyadd(df):
    x = df['x'].values
    y = df['y'].values
    x_min = df['x'].min() - 1
    x_max = df['x'].max() + 1
    y_min = df['y'].min() - 1
    y_max = df['y'].max() + 1
    
    y_add = np.array([i for i in np.arange(y_min,y_max)])
    x_add = np.array([x_min for i in np.arange(y_min,y_max)])
    x = np.append(x,x_add)
    y = np.append(y,y_add)
    y_add = np.array([i for i in np.arange(y_min,y_max)])
    x_add = np.array([x_max for i in np.arange(y_min,y_max)])
    x = np.append(x,x_add)
    y = np.append(y,y_add)
    x_add = np.array([i for i in np.arange(x_min,x_max)])
    y_add = np.array([y_min for i in np.arange(x_min,x_max)])
    x = np.append(x,x_add)
    y = np.append(y,y_add)
    x_add = np.array([i for i in np.arange(x_min,x_max)])
    y_add = np.array([y_max for i in np.arange(x_min,x_max)])
    x = np.append(x,x_add)
    y = np.append(y,y_add)
    
    XY = pd.DataFrame({'x':x,'y':y})
    
    return XY

def neighbourHist(tri):
    unique, counts = np.unique(tri.simplices, return_counts=True)
    nphist = np.histogram(counts,density = True,bins=range(1, 10))
    plt.plot(nphist[1][:-1],nphist[0], marker = "o", linestyle='none')
    plt.show()
    
def plotColorScatter(XY_red,var1,var2,var3,fileloc,vmax,vmin,colormap,alph,second):
    if vmax == "none":
        vmax = XY_red[var2].values.max()
    if vmin == "none":
        vmin = XY_red[var2].values.min()
    fig, ax = plt.subplots(figsize = (13, 10))
    cm = plt.cm.get_cmap(colormap)
    sc = plt.scatter(XY_red[var1].values,XY_red[var2].values, c = XY_red[var3].values, cmap=cm,vmax = vmax,vmin = vmin, alpha = alph)
    fig.colorbar(sc)
    plt.title("time = " + str(second))
    if fileloc != False:
        plt.savefig(fileloc)
    
    plt.show()
    
def plot_correlation(g_tensor, data_name,path,extention, r_array, phi_array, para, ow_bool):
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
    plt.savefig(path + file_name + extention)
    #latex_writer(file_name, caption, begin=True, end=False, overwrite=ow_bool)

    plt.show()
    plt.plot(r_array, mean_phi_g_tensor)

    plt.yscale('log')
    plt.savefig(path + file_name_log + extention)

    plt.show()
    #latex_writer(file_name_log, caption, begin=False, end=True, overwrite=False)

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
    plt.savefig(path + 'G_Heat_ ' + data_name + "_N_phi=_" + str(r_array.shape[0]) + extention)
    plt.show()
    


def labeler(label):
    if label == "Nd":
        return "<1/d> in $m^{-1}$"
    if label == "r":
        return "r in m"
    if label == "Bf":
        return "$\psi_6$"
    if label == "Dm":
        return "var(d)"
    if label == "speed_nn":
        return "v in m/s"
    if label == "angle":
        return "$\Theta$ in rad"
    if label == "dens":
        return "$\\rho$ in $m^2$"
    if label == "second":
        return "$t$ in s"
    return "unknown label"


def folderBuilder(path):
    os.system("mkdir " + path + "plots")
    os.system("mkdir " + path + "plots/structure")
    os.system("mkdir " + path + "plots/structure/XYcsv")
    os.system("mkdir " + path + "plots/structure/dist_factor")
    os.system("mkdir " + path + "plots/structure/bond_factor")
    
def folderCollector(folder_frame,T_test):
    path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()

    #folder_frame_frac = folder_frame.loc[folder_frame[test_str2] == T_test]
    folder_frame_frac = folder_frame.loc[folder_frame[test_str2] == T_test]['ini_folder'].values
    b_folder = folder_frame.loc[folder_frame[test_str2] == T_test]['b'].values
    loc_list = [[path + folder + "/" + "new_evac_traj_" + af.b_data_name(2 * bi, 3) + "_" + str(i) + ".txt" for i in
                 range(N_runs)] for folder, bi in zip(folder_frame_frac, b_folder)]
    return loc_list

def dfLoader(loc,second,fps,col):
    df = pd.read_csv(loc, sep="\s+", header=0, comment="#",skipinitialspace=True, usecols=col)
    if df['FR'].max() < second:
        print("max time passed")
        return 0
    XY = trajReduce(second,df,fps,True)
    return XY

def delaunayMaker(x,y):
    points = np.vstack((x,y)).T
    tri = Delaunay(points)
    return tri ,points

def XYLocalAppend(XY,second,test_str2,T_test,filtered,box):
    x = XY['x'].values
    y = XY['y'].values
    tri,points = delaunayMaker(x,y)
    N = tri.points.shape[0]
    neighbour_list, neigh_dist_list, local_bond_list, mean_neighdist_list = localOrientationMeasures(tri,N,filtered,box)
    XY['Nn'] = neighbour_list
    XY['Dm'] = neigh_dist_list/0.18
    XY['Bf'] = local_bond_list
    XY['Nd'] = mean_neighdist_list

    XY[test_str2] = listMaker(T_test,x.shape[0])
    XY['second'] = listMaker(second,x.shape[0])
    #XY_red = pedReducer(XY,x_min,x_max,y_min,y_max,300)
    return XY

def listMaker(value,length):
    list_ = np.empty(length)
    list_.fill(value)
    return list_

def DataFrameBuilder2(test,test2):
    path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()

    dfcsv = pd.read_csv(path + "plots/structure/XYcsv/filelist.csv")
    data = dfcsv[dfcsv[test_str2] == test2[0]]
    data = dfcsv[dfcsv['testvar'] == test[0]]["files"].values
    keys = pd.read_csv(data[0]).keys().values[1:]
    appenditures = np.array(["esigma","angle","r"])
    keys = np.append(keys,appenditures)
    print(keys)
    dfreturn = pd.DataFrame(columns=keys)
    dfs = []
    counter = 0
    for testvar2 in test2:
        for esigma in test:
            data = dfcsv[dfcsv['testvar'] == esigma]["files"].values
            #data = data[data['testvar'] in second_list]["files"].values
            for pathname,i in zip(data,range(data.shape[0])):
                dfread = pd.read_csv(pathname)
                length = len(dfread.index)
                esigmadf = listMaker(esigma,length)
                #runindexdf = listMaker(i,length)
                angle = np.arctan(dfread['x'].values/dfread['y'].values)
                radius = np.sqrt(dfread['x'].values ** 2 + dfread['y'].values ** 2)
                
                appendix = [esigmadf,angle,radius]
                for append,name in zip(appendix,appenditures):
                    dfread[name] = append
                dfs.append(dfread)
                
                print(counter, "/", len(test2) * len(test) * data.shape[0])
                #dfreturn = dfreturn.append(dfread)
                counter += 1

                #print(dfread.head())
    print(dfs[-1]['second'])
    dfreturn = pd.concat(dfs)
    
    return dfreturn

def timeCorrelationMatrix(lattice_x,lattice_y,x_array,y_array,a):
    x_dens = np.array(
        [lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])

    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix

def correlationCalculator(cormatrix,tau,second,time,x,y,x_array,y_array,a):
    latticeX_t = df[df[time] == int(second)][x]
    latticeY_t = df[df[time] == int(second)][y]
    latticeX_tau = df[df[time] == int(second + tau)][x]
    latticeY_tau = df[df[time] == int(second + tau)][y]
    matrix_t = timeCorrelationMatrix(latticeX_t,latticeY_t,x_array,y_array,a)
    matrix_tau = timeCorrelationMatrix(latticeX_tau,latticeY_tau,x_array,y_array,a)
    matrix_t_tau = matrix_t * matrix_tau
    cormatrix += matrix_t_tau

    return cormatrix

def timeCorrelation(df,x,y,time,timelist,taulist,x_array,y_array,a):
    timelist = np.array(timelist)
    taulist = np.array(taulist)
    correlation = np.empty(int(taulist.shape[0]))
    
    for tau,i in zip(taulist, range(taulist.shape[0])):
        print("<<<<<<<<<<<< tau = ", tau, " >>>>>>>>>>>>")
        cormatrix = np.zeros((x_array.shape[0],y_array.shape[0]))
        pool = Pool()
        g_pool = np.array([pool.apply_async(correlationCalculator, args=(cormatrix,tau,second,time,x,y,x_array,y_array,a)) for second in timelist])
        
        cormatrix_collection = np.array([p.get() for p in g_pool])
        print(cormatrix_collection.shape)
        cormatrix = cormatrix_collection.sum(axis=1)
        pool.close()
        correlation[i] = cormatrix.mean()
    return correlation/timelist.max()

def plotter(df,esigmas,test2,plotinfo,factor,xscale,binval,bins,inter,color,error_bar,teststr2,var2):
    fig, ax = plt.subplots(figsize = (9, 6))
    for t2 in test2:
        df_t2 = df[df[test_str2] == t2]
        for esigma in esigmas:
            df_sigma = df_t2[df_t2['esigma'] == esigma]
            #df_sigma = df_sigma[df_sigma['r'] > 1]
            minval = df_sigma[binval].min()
            maxval = df_sigma[binval].max()
            print(minval,maxval)
            df_sigma[bins] = pd.cut(x = df_sigma[binval], bins=np.arange(minval,maxval,inter), labels=[i for i in np.arange(minval,maxval - inter,inter)])
            #df_sigma = df_sigma[df_sigma['r'] < 5]
            
            plot = df_sigma.groupby([bins]).mean()
            yerr = df_sigma.groupby([bins]).std()
            #print(df_sigma)
            if binval == xscale:
                xplot = plot[binval].index
            else:
                xplot = plot[xscale]
            if error_bar:

                plt.errorbar(xplot,plot[factor],yerr[factor],marker='o',linestyle = "none",label = "$\sigma$ = " + str(t2))
            else:
                if color:
                    #plt.scatter(xplot,plot[factor],c = plot[binval],label = "$\sigma$ = " + str(esigma))
                    cm = plt.cm.get_cmap("plasma")
                    sc = plt.scatter(xplot,plot[factor], c = plot[binval] , cmap=cm)
                    fig.colorbar(sc,ax = ax)
                else:
                    return plt.scatter(xplot,plot[factor],label = "$\sigma$ = " + str(esigma))
                    

    plt.legend()
    plt.xlabel(labeler(xscale))
    plt.ylabel(labeler(factor))
    plt.title("time = " + str(plotinfo[6]))
    if plotinfo[4]:
        plt.xlim(plotinfo[2])
    if plotinfo[5]:
        plt.ylim(plotinfo[3])
    plt.yscale(plotinfo[1])
    plt.xscale(plotinfo[0])
    
    plt.savefig(path + "plots/structure/" + factor+xscale+bins+teststr2+str(var2)+str(plotinfo[6]) + plotinfo[7])
    plt.show()
    
def png_name(num,length):
    string = str(num)
    while len(string) < length:
        string = "0" + string
    return("img" + string)
    
def orderfilter(df_new,second,order,i,superior):
    df_i = df_new[df_new['i'] == i]
    df_i = df_i[df_i['second'] == second]
    if superior:
        df_i = df_i[df_i['Bf'] >= order]
    else:
        df_i = df_i[df_i['Bf'] < order]
    return df_i['id'].values
    
def connectedNeighbours(df_id,x,y,threshold,precision):
    distance = np.round(np.sqrt((df_id['x'].to_numpy() - x)**2 + (df_id['y'].to_numpy() - y)**2),precision)
    df_id['dist'] = distance
    df_connected = df_id[df_id['dist'] < threshold]
    return df_connected

def ripsfiltration(df_id,id_list,threshold,precision):
    G = nx.Graph()
    for i in id_list:
        dfi = df_id[df_id['id'] == i]
        x = dfi['x'].to_numpy()[0]
        y = dfi['y'].to_numpy()[0]
        G.add_node(i,pos = (x,y))
        df_connected = connectedNeighbours(df_id,x,y,threshold,precision)
        for j, x, y in zip(df_connected['id'], df_connected['x'], df_connected['y']):
            if j != i:
                G.add_node(j,pos = (x,y),weight = 1)
                G.add_edge(i,j)
    return G

def OrderNeighDistDiv(df,graphs,id_list,minsec,tmax,d):
    from scipy.signal import savgol_filter

    df_id = df[df['id'].isin(id_list)]
    df_id = df_id[df_id['second'] > minsec]
    #print(id_list)
    nd_array = np.empty(0)
    dy_array = np.empty(0)
    bf_array = np.empty(0)
    for ids in graphs:
        df_id_i = df_id[df_id['id'].isin(ids)]
        if df_id_i['second'].shape[0] > 51:
            bf_group = df_id_i.groupby("second")['Bf'].mean()
            nd_group = df_id_i.groupby("second")['Nd'].mean()
            df_newgroup = pd.DataFrame({"time":bf_group.index.to_numpy(),"Bf":bf_group.to_numpy(),"Nd":nd_group.to_numpy()})
            maxval = df_newgroup['time'].max()
            minval = df_newgroup['time'].min()
            inter = 15
            df_newgroup['t0'] = pd.cut(x = df_newgroup['time'],
                                       bins=np.arange(minval,maxval,inter), labels=[i for i in np.arange(minval,maxval - inter,inter)])
            print(df_newgroup['time'].shape,df_newgroup['Bf'].shape,df_newgroup['Nd'].shape,df_newgroup['t0'].shape)

            plot = df_newgroup.groupby(['t0']).mean()
            if plot['Bf'].shape[0] > 51:
                #plot = plot[plot['Nd'] - 0.4 < 0.07]
                yhat = savgol_filter(plot['Bf'], 21, 8)
                derivative = np.gradient(yhat)
                print(plot.index.shape,derivative.shape,plot['Nd'].to_numpy().shape,plot['Bf'].to_numpy().shape)

                #df_dy = pd.DataFrame({'time':plot.index,'dy': derivative,"Nd":plot['Nd'].to_numpy() - d,"Bf":plot['Bf'].to_numpy()})
                nd_array = np.append(nd_array,plot['Nd'].to_numpy() - d)
                bf_array = np.append(bf_array,plot['Bf'].to_numpy())
                dy_array = np.append(dy_array,derivative)
            
            plt.plot(plot.index,plot['Bf'],marker = "o")
            plt.plot(plot.index,plot['Nd'] - d,marker = "o")
            plt.plot(plot.index,yhat)
            plt.plot()
            plt.ylim([0,1])
            plt.xlim([0,tmax])
            plt.show()
    df_link = pd.DataFrame({"nd":nd_array,"dy":dy_array, "bf":bf_array})
    return df_link

def linkplot(df_link,inter):

    maxval = df_link['nd'].max()
    minval = df_link['nd'].min()
    print(minval,maxval)
    df_link['nd0'] = pd.cut(x = df_link['nd'],
                            bins=np.arange(minval,maxval,inter), labels=[i for i in np.arange(minval,maxval - inter,inter)])
    plot = df_link.groupby('nd0').mean()
    ploterr = df_link.groupby('nd0').std()
    return plot,ploterr

def derivativeTrajectory(id_list,df_i,df,threshold,precision,mintime,d,t_max):
    df_id = df_i[df_i['id'].isin(id_list)]
    print("df_id shape = ",df_id.shape)
    df_id = df_id[df_id['second'] == t_max]
    print("df_id tmax shape = ",df_id.shape)

    sorte = df_id.sort_values(['id'], ascending = [True])
    df_id = sorte.groupby('id').first().reset_index()
    print("<Graph>")
    G = ripsfiltration(df_id,id_list,threshold,precision)
    print("</Graph>")
    graphs = list(nx.connected_component_subgraphs(G))
    print(len(graphs))
    print("<df_link>")
    df_link = OrderNeighDistDiv(df,graphs,id_list,mintime,t_max,d)
    print("</df_link>")

    return df_link

def frameOrderFilter(df_i,filtertime,threshold,i,supremumfilter):
    id_list = sf.orderfilter(df_i,filtertime,threshold,i,supremumfilter)
    df_filtered = df_i[df_i['id'].isin(id_list)]

    return df_filtered


