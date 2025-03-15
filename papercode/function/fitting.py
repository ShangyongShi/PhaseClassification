#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:52:59 2023

@author: sshi
"""

import matplotlib.pyplot as plt
import numpy as np

def get_contour_values(df):
    '''
    cn is the returned contour 
    '''
    cn = plt.contour(df.index, df.columns, df.T, levels=np.arange(0, 1.1, 0.1))
    plt.close()
    contours = []
    idx = 0
    # match contour value with corresponding coordinates
    for cc, vl in zip(cn.collections, cn.levels):
        for pp in cc.get_paths():
            paths = {}
            paths['id'] = idx
            paths["value"] = float(vl)
            
            v = pp.vertices
            x = v[:, 0]
            y = v[:, 1]
            
            paths["x"] = x
            paths["y"] = y
            contours.append(paths)
            idx += 1
            
    # merge arrays with the same contour value
    xs, ys = {}, {}
    for value in cn.levels:
        x, y = [], []
        for contour in contours:
            if contour['value'] == float(value):
                x+=list(contour['x'])
                y+=list(contour['y'])
        
        xy = np.array([x, y])
        xy = xy[:, xy[0,:].argsort()]
        
        key = np.round(value, decimals=1)
        if np.size(xy)>0:  
            xy = np.unique(xy, axis=1)
        
            xs[key] = xy[0, :]
            ys[key] = xy[1, :]
        else:
            xs[key], ys[key] = [], []
    return xs, ys