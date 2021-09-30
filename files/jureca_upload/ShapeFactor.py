import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math as m
from multiprocessing import Pool
import random as rd
from pathlib import Path
import sys
from scipy.integrate import simps
import itertools
import AnalFunctions as af
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def ShapeFactor(lat,room):
    if lat.shape[0] == 0:
        return
    vor = Voronoi(lat)
    vert = vor.regions
    rig_vert = []
    poly_room = room_geo(room)

    for note in vert:
        if -1 in note:
            continue
        rig_vert.append(note)

    pol_area_list = []
    pol_perimeter_list = []
    for note in rig_vert[1:]:
        coords = [(vor.vertices[i][0], vor.vertices[i][1]) for i in note]
        poly = Polygon(coords)
        if poly.centroid.within(poly_room):
            pol_area_list.append(poly.area)
            pol_perimeter_list.append(poly.length)
    return np.array(pol_perimeter_list) ** 2 / (4 * np.pi * np.array(pol_area_list))

def room_geo(box):
    #coords = [(-0.25,-1), (-0.25,-0.15),(-0.4,0),(-b/2,0),(-b/2,5),(b/2,5),(b/2,0),(0.4,0), (0.25,-0.15),(0.25,-1)]
    coords = [(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0],box[3])]

    poly_room = Polygon(coords)
    return poly_room
