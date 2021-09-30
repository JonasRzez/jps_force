
from data_ini import dataFrameInitilizer
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
import ShapeFactor as shape
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
import cmath
from ripser import ripser
from persim import plot_diagrams
from ripser import Rips
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon
from scipy.stats import moment
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()

def widthOfGaußian(fwhm):
    return fwhm * np.sqrt(2) / (2 * m.sqrt(2 * np.log(2)))

def densty1d(delta_x, a):
    return np.array(list(map(lambda x: 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2), delta_x)))

def densty1dWeight(delta_x, weight,a):
    return np.array(list(map(lambda var: var[1] / (m.sqrt(m.pi) * a) * m.e ** (-var[0] ** 2 / a ** 2), zip(delta_x,weight))))

def densityField(x_dens, y_dens, a):
    #x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    #y_dens = np.array([lattice_y - y for y in y_array])

    #rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    #rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    #rho_matrix_x = densty1d(x_dens, a)
    #rho_matrix_y = densty1d(y_dens, a)
    
    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.T

def xdensYdens(lattice_x,lattice_y,x_array,y_array):
    x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])
    return x_dens,y_dens

def orderField(x_dens,y_dens,order,a):
    #x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    #y_dens = np.array([lattice_y - y for y in y_array])
    #rho_matrix_x = densty1d(x_dens, a)  # density matrix is calculated
    #rho_matrix_y = densty1dWeight(y_dens, order, a)
    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1dWeight(delta_y,order, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    order_matrix = rho_matrix.T#/densityField(x_dens, y_dens, a)
    
    return order_matrix

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

def widthOfGaußian(fwhm):
    return fwhm * np.sqrt(2) / (2 * m.sqrt(2 * np.log(2)))

def densty1d(delta_x, a):
    return np.array(list(map(lambda x: 1 / (m.sqrt(m.pi) * a) * m.e ** (-x ** 2 / a ** 2), delta_x)))

def densty1dWeight(delta_x, weight,a):
    return np.array(list(map(lambda var: var[1] / (m.sqrt(m.pi) * a) * m.e ** (-var[0] ** 2 / a ** 2), zip(delta_x,weight))))

def densityField(x_dens, y_dens, a):
    #x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    #y_dens = np.array([lattice_y - y for y in y_array])

    #rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    #rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    #rho_matrix_x = densty1d(x_dens, a)
    #rho_matrix_y = densty1d(y_dens, a)
    
    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1d(delta_y, a) for delta_y in y_dens])
    
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.T

def xdensYdens(lattice_x,lattice_y,x_array,y_array):
    x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    y_dens = np.array([lattice_y - y for y in y_array])
    return x_dens,y_dens

def orderField(x_dens,y_dens,order,a):
    #x_dens = np.array([lattice_x - x for x in x_array])  # calculate the distant of lattice pedestrians to the measuring lattice
    #y_dens = np.array([lattice_y - y for y in y_array])
    #rho_matrix_x = densty1d(x_dens, a)  # density matrix is calculated
    #rho_matrix_y = densty1dWeight(y_dens, order, a)
    rho_matrix_x = np.array([densty1d(delta_x, a) for delta_x in x_dens])  # density matrix is calculated
    rho_matrix_y = np.array([densty1dWeight(delta_y,order, a) for delta_y in y_dens])
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    order_matrix = rho_matrix.T#/densityField(x_dens, y_dens, a)
    
    return order_matrix

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
            
"""def plotter(df,esigmas,test2,plotinfo,factor,xscale,binval,bins,inter,color,error_bar,label):
    for t2 in test2:
        df_t2 = df[df[test_str2] == t2]
        for esigma in esigmas:
            df_sigma = df_t2[df_t2['esigma'] == esigma]
            df_sigma = grouper(df_sigma,binval,bins,inter)
            
            plot = df_sigma.groupby(bins).mean()
            #print(df_sigma)
            if binval == xscale:
                xplot = plot[binval].index
            else:
                xplot = plot[xscale]
            if error_bar:
                plot = df_sigma.groupby(bins)[factor].mean()
                p025 = df_sigma.groupby(bins)[factor].quantile(0.025)
                p975 = df_sigma.groupby(bins)[factor].quantile(0.975)
                yerr = [plot - p025, plot - p025]
                plt.errorbar(plot.index,plot,yerr,marker = "o",linestyle = "none")
            else:
                if color:
                    #plt.scatter(xplot,plot[factor],c = plot[binval],label = "$\sigma$ = " + str(esigma))
                    cm = plt.cm.get_cmap("plasma")
                    sc = plt.scatter(xplot,plot[factor], c = plot[binval] , cmap=cm)
                    fig.colorbar(sc,ax = ax)
                else:
                    return plt.scatter(xplot,plot[factor],label = label)"""

def plotter(df,esigmas,test2,plotinfo,factor,xscale,binval,bins,inter,color,error_bar,label):
    for t2 in test2:
        print("t2 = ", t2)
        df_t2 = df[df[test_str2] == t2]
        for esigma in esigmas:
            df_sigma = df_t2[df_t2['esigma'] == esigma]
            df_sigma = grouper(df_sigma,binval,bins,inter)
            
            plot = df_sigma.groupby([bins]).mean()
            yerr = df_sigma.groupby([bins]).std()
            #print(df_sigma)
            if binval == xscale:
                xplot = plot[binval].index
            else:
                xplot = plot[xscale]
            if error_bar:

                plt.errorbar(xplot,plot[factor],yerr[factor],marker='o',linestyle = "none",label = label)
            else:
                if color:
                    #plt.scatter(xplot,plot[factor],c = plot[binval],label = "$\sigma$ = " + str(esigma))
                    cm = plt.cm.get_cmap("plasma")
                    sc = plt.scatter(xplot,plot[factor], c = plot[binval] , cmap=cm)
                    fig.colorbar(sc,ax = ax)
                else:
                    return plt.scatter(xplot,plot[factor],label = label)
    
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
    
def orderfilter(df_new,filtervalue,lb,hb,order,i,superior):
    df_i = df_new[df_new['i'] == i]
    df_i = df_i[df_i[filtervalue] > lb]
    df_i = df_i[df_i[filtervalue] < hb]

    if superior:
        df_i = df_i[df_i['Bf'] >= order]
    else:
        df_i = df_i[df_i['Bf'] < order]
    return df_i['id'].values

def putter(counter,filler,filling,count_up = True):
    arange = np.arange(counter,counter + filling.shape[0])
    np.put(filler,arange,filling)
    if count_up:
        counter += filling.shape[0]
        return counter
    return

def grouper(df_bin,binval,bins,inter):
    minval = df_bin[binval].min()
    maxval = df_bin[binval].max()
    #print(minval,maxval)
    interval_range = np.arange(minval,maxval + inter,inter)

    lables = [i for i in interval_range]
    #print(lables)
    bins_array = np.arange(minval-inter,maxval+inter,inter)
    #print(bins_array)
    df_bin[bins] = pd.cut(x = df_bin[binval], bins=bins_array, labels=[i for i in interval_range])
    return df_bin

def filtrationplot(df_g,graphs,color,size):
    for g,col in zip(graphs,color):
        graph_list = []
        """if len(g) > size:
            graph_list.append(list(g))
            graph_len.append(len(list(g)))"""
        df_cg = df_g[df_g['id'].isin(g)]
        ax.scatter(df_cg['x'], df_cg['y'],c = col)

def ShapeFactor(lat,room):
    if lat.shape[0] == 0:
        print("WARING: lat.shape[0] = 0")
        return
    vor = Voronoi(lat,qhull_options='Qbb Qc Qx')
    vert = vor.regions
    rig_vert = []
    poly_room = shape.room_geo(room)
    centerpoints = []
    for note,centerpoint,i in zip(vert,vor.points,range(vor.points.shape[0])):
        if -1 in note:
            continue
        rig_vert.append(note)
        ki = np.where(vor.point_region == i)[0][0]
        centerpoints.append(vor.points[ki])

    pol_area_list = []
    pol_perimeter_list = []
    pol_centroid_x = []
    pol_centroid_y = []
    for note in rig_vert:
        coords = [(vor.vertices[i][0], vor.vertices[i][1]) for i in note]
        poly = Polygon(coords)
        if poly.centroid.within(poly_room):
            pol_area_list.append(poly.area)
            pol_perimeter_list.append(poly.length)
            pol_centroid_x.append(poly.centroid.x)
            pol_centroid_y.append(poly.centroid.y)

    return np.array(pol_perimeter_list) ** 2 / (4 * np.pi * np.array(pol_area_list)),np.array(centerpoints)[:,0],np.array(centerpoints)[:,1]

def addPedFrame(box):
    Nx = int((box[1] - box[0]) * 10)
    Ny = int((box[3] - box[2]) * 10)
    x_frame = np.linspace(box[0],box[1],Nx)
    x_frame = np.append(x_frame, np.linspace(box[0],box[1],Nx))
    x_frame = np.append(x_frame, sf.listMaker(box[0],Ny))
    x_frame = np.append(x_frame, sf.listMaker(box[1],Ny))
    
    y_frame = sf.listMaker(box[2],Nx)
    y_frame = np.append(y_frame,sf.listMaker(box[3],Nx))
    y_frame = np.append(y_frame,np.linspace(box[2],box[3],Ny) )
    y_frame = np.append(y_frame,np.linspace(box[2],box[3],Ny))
    
    return np.array([np.array([xi, yi]) for xi, yi in zip(x_frame, y_frame)])

def groupPlot(df_total,xval,yval,ax_i,inter,error,label = " "):
    df_grouped = grouper(df_total,xval,"val_0",inter)
    plot = df_grouped.groupby(["val_0"]).mean()
    
    xplot = plot[xval]
    if error:
        val_0_sorted = np.sort(np.unique(df_grouped.val_0.to_numpy()))
        error_list025 = []
        error_list075 = []

        for val_0,ymean in zip(val_0_sorted,plot[yval]):
            #print(val_0)
            df_g = df_grouped[df_grouped.val_0 == val_0]

            error_list025.append(ymean - np.quantile(df_g[yval],0.15))
            error_list075.append(np.quantile(df_g[yval],0.75) - ymean)

            #print(interval95)
        errors = np.array([error_list025[1:],error_list075[1:]])

        #print(errors)
        #plt.scatter(plot[xval],plot[yval])
        #print([error_list025,error_list075][2:])
        ax_i.errorbar(
            plot[xval][1:],
            plot[yval][1:],
            yerr = errors,
            marker = "o",
            linestyle = "none",
            label = label)
    else:
        ax_i.scatter(xplot,plot[yval],label = label)

def densNormCalc(df_r,t_list,alpha):
    i_list = np.unique(df_r.i.to_numpy())
    print(i_list)
    mean_dens_list = []
    dens_normal = np.empty(df_r.shape[0])
    counter = 0
    for i in i_list:
        #print(i)
        df_i = df_r[df_r['i'] == i]
        #df_i = df
        #df_i = df_i.sort_values(by = ['second','x','y'])
        for t in t_list:
            df_t = df_i[df_i['second'] == t]
            #print(df_t.shape)
            #df_t = df_t.sort_values(by = ['x','y'])
            #print(df_t['x'].to_numpy(),df_t['y'].to_numpy())
            length = df_t.shape[0]
            #mean_dens = df_t['dens'].to_numpy().mean()
            #std_dens = df_t['dens'].to_numpy().std()

            mean_dens = df_t['dens'].to_numpy().mean()
            std_dens = df_t['dens'].to_numpy().std()

            mean_dens_array = ((df_t.dens.to_numpy() - mean_dens)   / std_dens) * (df_t.dens.to_numpy()/rho_max) ** alpha

            #mean_dens_array = (df_t['dens'].to_numpy() - mean_dens)/std_dens
            counter =  putter(counter,dens_normal,mean_dens_array)
    df_r['dens_norm'] = dens_normal
    return df_r

def dfRsegmenter(df_seg,r_min,r_max,r_jump,t_list,alpha):
    df_r_list = []
    r_list = np.arange(r_min,r_max,r_jump)
    for r in r_list:
        df_r = df_seg[df_seg.r > r]
        df_r = df_r[df_r.r <= r + r_jump]
        df_r = densNormCalc(df_r,t_list,alpha)
        df_r_list.append(df_r)
    return df_r_list

def dfMeanDensityAtTimeMaker(df_r_list,t_array):
    mean = []
    for df_r in df_r_list:
        mean_r = []
        for t in t_array:
            df_tr = df_r[df_r['second'] == t]
            mean_r.append(round(df_tr.dens.mean(),2))
        mean.append(mean_r)
    
    map_df = {'t':t_array}
    for m,z in zip(mean,range(len(mean))):
        map_df['z' + str(z)] = m
    df_mean = pd.DataFrame(map_df)
    return df_mean

def timeListEqualDensityMaker(df_mean,mean_list,meanint):
    
    df_mean_keys = df_mean.keys().to_numpy()[1:]
    time_list = []

    for ml in mean_list:
        print(ml)
        time_list_i = []
        for key in df_mean_keys:
            df_filter = df_mean[df_mean[key] > ml]
            time_filter = df_filter[df_filter[key] < ml + meanint]['t'].to_numpy()
            if time_filter.shape[0] > 0:
                time_list_i.append(df_filter[df_filter[key] < ml + meanint]['t'].to_numpy()[0])
            else:
                print("WARNING: density is nan")

        time_list.append(time_list_i)
    return time_list

def dfMeanDensityCombine(df_r_list,t_ml):
    df_tr_list = []
    for df_r,ti in zip(df_r_list,t_ml):
        df_tr = df_r[df_r.second == ti]
        df_tr_list.append(df_tr)
    df_total = pd.concat(df_tr_list)
    return df_total
    
def VoronoiRidges(vor,ax):
    lines = []
    for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                #print(vor.vertices[simplex, 0])
                lines.append([(vor.vertices[simplex, 0][0],vor.vertices[simplex, 1][0]) , (vor.vertices[simplex, 0][1],vor.vertices[simplex, 1][1])])
                #x.append(vor.vertices[simplex, 0])
                #y.append(vor.vertices[simplex, 1])


    #lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    #print(lines)
    lc = LineCollection(lines,color = 'black')
    ax.add_collection(lc)
    ax.autoscale()
    
def frameOrderFilter(df_i,filtertime,threshold,i,supremumfilter):
    id_list = sf.orderfilter(df_i,filtertime,threshold,i,supremumfilter)
    df_filtered = df_i[df_i['id'].isin(id_list)]
    
    return df_filtered

def PlotHistogramHeatMap(df_h,distvalue,dimension,dr,bin_array,r_array,ax_i,colormap,max_scale):
    DensityHeatMatrix = np.empty([r_array.shape[0],bin_array[:-1].shape[0]],dtype = float)
    for r,i in zip(r_array,range(r_array.shape[0])):
        df_t_r = df_h[(df_h[dimension] > r) & (df_h[dimension] < r + dr) ]
        #print(df_t_r.shape)
        n,bins = np.histogram(df_t_r[distvalue], bins = bin_array, density = True)
        DensityHeatMatrix[i,:] = n
    vmax = DensityHeatMatrix.max()/max_scale
    vmin = DensityHeatMatrix.min()
    #print(vmin,vmax)
    ax_i.pcolormesh(bin_array, r_array, DensityHeatMatrix, cmap = colormap,vmin = vmin, vmax = vmax)
    
def add_wall(line_width,bi,ax):
    wall1 = plt.Line2D((-bi / 2, -bi / 2), (0, 100), lw=line_width)
    wall2 = plt.Line2D((bi / 2, bi / 2), (0, 100), lw=line_width)
    wall3 = plt.Line2D((-bi / 2, -0.45), (0, 0), lw=line_width)
    wall4 = plt.Line2D((0.45, bi / 2), (0, 0), lw=line_width)
    wall5 = plt.Line2D((0.45, 0.25), (0., -0.15), lw=line_width)
    wall6 = plt.Line2D((-0.45, -0.25), (0., -0.15), lw=line_width)
    wall7 = plt.Line2D((0.25, 0.25), (-0.15, -1.0), lw=line_width)
    wall8 = plt.Line2D((-0.25, -0.25), (-0.15, -1.), lw=line_width)
    ax.add_line(wall1)
    ax.add_line(wall2)
    ax.add_line(wall3)
    ax.add_line(wall4)
    ax.add_line(wall5)
    ax.add_line(wall6)
    ax.add_line(wall7)
    ax.add_line(wall8)

def EvacuationPlot(sec,df_i,x_min,x_max,y_min,y_max,ax):
    df_ti_i = df_i[df_i['second'] == sec]
    cm = plt.cm.get_cmap('plasma')
    add_wall(2.5,50,ax)
    for x,y in zip(df_ti_i['x'],df_ti_i['y']):
        circ = plt.Circle((x,y), radius=0.2, color='grey',alpha = 0.5)
        ax.add_patch(circ)
    #ax.scatter(df_ti_i['x'],df_ti_i['y'], c = df_ti_i['Bf'], cmap=cm,vmax = 1,vmin = 0, alpha = 1)
    ax.scatter(df_ti_i['x'],df_ti_i['y'])
    
def groupValPlot(df_total,groupval,xval,yval,ax_i,inter,error,color,colormap,label = ' '):
    df_grouped = grouper(df_total,groupval,"val_0",inter)
    df_grouped = df_grouped[df_grouped.r > 1]
    #print(df_grouped.sort_values(by = "val_0"))
    plot = df_grouped.groupby(["val_0"]).mean()
    xplot = plot[xval]
    if error:
        val_0_sorted = np.sort(np.unique(df_grouped.val_0.to_numpy()))
        error_list025 = []
        error_list075 = []

        for val_0,ymean in zip(val_0_sorted,plot[yval]):
            df_g = df_grouped[df_grouped.val_0 == val_0]
            error_list025.append(ymean - np.quantile(df_g[yval],0.025))
            error_list075.append(np.quantile(df_g[yval],0.975) - ymean)
        errors = np.array([error_list025[1:],error_list075[1:]])
        ax_i.errorbar(
            plot[xval][1:],
            plot[yval][1:],
            yerr = errors,
            marker = "o",
            linestyle = "none",
            label = label)
    else:
        if color:
            print(plot.index.min(),plot.index.max())
            ax_i.scatter(xplot,plot[yval], c = plot[groupval],cmap = colormap, vmin = plot.index.min(), vmax = plot.index.max(), label = label)
        else:
            ax_i.scatter(xplot,plot[yval],label = label)
            
def PlotHistogramHeatMap(df_h,distvalue,dimension,dr,bin_array,r_array,ax_i,colormap,max_scale,ranges = True):
    DensityHeatMatrix = np.empty([r_array.shape[0],bin_array[:-1].shape[0]],dtype = float)
    for r,i in zip(r_array,range(r_array.shape[0])):
        if ranges:
            df_t_r = df_h[(df_h[dimension] > r) & (df_h[dimension] < r + dr)]
        else:
            df_t_r = df_h[(df_h[dimension] == r)]
        #print(df_t_r.shape)
        n,bins = np.histogram(df_t_r[distvalue], bins = bin_array, density = True)
        DensityHeatMatrix[i,:] = n
    #print(n,bins)
    vmax = DensityHeatMatrix.max()/max_scale
    vmin = DensityHeatMatrix.min()
    #print(vmin,vmax)
    ax_i.pcolormesh(bin_array, r_array, DensityHeatMatrix, cmap = colormap,vmin = vmin, vmax = vmax)
    
def HistoPlot(data,bins_array,ax_i,color,marker,label):
    n,bins = np.histogram(data,bins = bins_array,density = True)
    line, = ax_i.plot((bins[:-1] + bins[1:])/2,n,color = color,marker = marker,linestyle = "none",label = label)
    return line

def imgName(i):
    if i < 10:
        return "00" + str(i)
    if i >= 10 and i < 100:
        return "0" + str(i)
    if i < 1000:
        return str(i)

def filelistWriter(load,li,lf,index_bool,ind,time_list):
    
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
                        #else:
                            #sf.orderfetch(second,ni,df,fps,test_str2,T_test,csvname,True,box)
                        
                    #else:
                        #sf.orderfetch(second ,ni,df,fps,test_str2,T_test,csvname,True,box)
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
    
    
    
load = True
esigmas = lin_var[test_var]
print(esigmas)
"""li = 1 #points in test_list
lf = li + 1"""
li = 0 #points in test_list
lf = 1
index_bool = True
range_max = 30
ind = np.arange(0,range_max)
print(ind)
t_max = 2200
t_min = 0
t_int = int((t_max - t_min)/61)
t_jump = 10

#time_list = np.arange(0,1000,50)

#time_list = np.arange(0,1000,10)
#time_list = np.append(time_list,np.arange(1000,11000,50))
#time_list = np.arange(0,10000,100)
time_list = np.arange(0,100,1)
#time_list = [350,450,500,600]
print(time_list)
#time_list = np.arange(0,1000,50)
#time_list = np.append(time_list,np.arange(1500,11000,500))
#time_list = np.array([100,400,750,1060,1350,1700,2000,4000,7000,10000])
#time_list = np.append(time_list,np.arange(1000,11000,50))
#print(time_list)
#time_list = [1440,1506,1542,1604,1640,1728,1830,1928,2082,2190]
#time_list = time_list[::10]

df,df_list = dataFrameInitilizer(load,li,lf,index_bool,ind,time_list)

df_new = sf.pedReducer(df,-10,10,0.,25,50)

i = 0
df_i = df[df.i == i]
#id_list = df_dist[(df_dist.dist > 0.75) & (df_dist.i == i)].id

t = time_list[-1]
th = 0.88
#[10,50,100,400,1400]
#time_list = time_list[time_list > 2000]
#fig, axis = plt.subplots(2,4,figsize=(5 * 4 ,5 * 2))
for esig in np.unique(df_i.esigma.to_numpy()):
    plotpath = "texfiles/16_09_2021_videos/testvideo_"
    os.system("mkdir " + plotpath)
    
    print(plotpath)

    df_esig = df_i[df_i.esigma == esig]
    print(print("\sigma^2 = ", esig))
    print(time_list)
    for t,i in zip(np.unique(df_esig.second.to_numpy()),range(np.array(time_list).shape[0])):
        df_t = df_esig[df_esig.second == t]
        df_57 = df_t[df_t.Nn != 6]
        print(df_t.shape,t)
        #df_6 = df_t[df_t.Nn == 6]
        #df_57 = df_t[df_t.Nn != 6]
        #df_id = df_t[df_t.id.isin(id_list)]

        x = df_t.x.to_numpy()
        y = df_t.y.to_numpy()
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()

        room = [x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3]
        pointframe = addPedFrame([x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
        lat = np.vstack((x,y)).T
        lat = np.vstack([lat,pointframe])
        vor = Voronoi(lat,qhull_options='Qbb Qc Qx')
        fig, ax = plt.subplots(1,figsize = (10 * 1., 10 * 10/20))

        VoronoiRidges(vor,ax)
        b
        cmap = cm.get_cmap('plasma')
        shapescatter = ax.scatter(df_t.x,df_t.y,color = "grey")
        #df_t = df_t[df_t.Bf >= 0.8]
        shapescatter = ax.scatter(df_t.x,df_t.y,c=df_t.Bf, vmin=0, vmax=1, cmap = cmap,alpha = 1.)
        #shapescatter = ax.scatter(df_t.x,df_t.y,c=df_t.ShapeFactor, vmin=1.05, vmax=1.3, cmap = cmap,alpha = 1.)
        #sc = ax.scatter(df_t.x,df_t.y,color="grey")

        #sc = ax.scatter(df_57.x,df_57.y,c=df_57.Nn, vmin=3., vmax=9., cmap = cm.get_cmap('PiYG', 5),alpha = 1.)

        ax.set_xlim([-10,10])
        add_wall(2.5,50,ax)
        ax.set_ylim([-1,10])
        ax.set_title("time = " + str(t) + " s")

        #fig.colorbar(sc, ticks = [4,5,6,7,8], label = "Coordination Number")
        #fig.colorbar(shapescatter, label = "$\\zeta$")

        fig.savefig(plotpath + "/" + imgName(i) + ".png")
        #fig.savefig(plotpath + "/" + af.b_data_name(esig,3) + ".png")
        plt.close()
        #plt.show()
        #break
