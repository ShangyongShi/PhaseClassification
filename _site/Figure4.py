# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:51:41 2023

@author: ssynj
"""
from function.conp import *


import sympy as sp
from sympy import solve, symbols
from scipy.optimize import curve_fit



def fit_50_contour(conp):
    contours = measure.find_contours(conp.values, 0.5)[0]
    X, Y = contours[:, 0], contours[:, 1]
    
    X, Y = select_data_points_for_fitting( conp)
    popt = fit_with_exp(X, Y)
    return popt, X, Y

## -------sub functions: 
def get_50_contour_values(pc):
    # input: contourf object
    # get the values of the contour line produced by Matplotlib
    X, Y = pc.collections[4].get_paths()[0].vertices.T
    idx = X.argsort()
    c50 = Y[idx]
    return c50

def monoExp(x, m, t, b):
    return m * np.exp(t * x +b) 

def select_data_points_for_fitting( conp):
    
    contours = measure.find_contours(conp.values, 0.5)[0]
    X, Y = contours[:, 0], contours[:, 1]
    
    # first column is x axis, second is y, and these are indexes that need to be 
    # broadcasted into original scale.

    # to select several data points from each bin for the function fitting
    binw = 0.2
    
    nbin = len(np.arange(0, 2+binw, binw))
    
    X = np.array(
                 X.tolist()+  
                 [6.7] +
                 [7.2] +
                 [8])
    
    Y = contours[:, 1]
    
    Y = np.array(Y.tolist() + 
                 [0.]  +
                 [0.]  +
                 [0.] )
    Y[0] = 1.32
    return X, Y

def fit_with_exp(X, Y):
    # Fit the contour with exp function
    from sklearn.metrics import r2_score
    popt, pcov = opt.curve_fit(monoExp, X, Y)
    y_pred = monoExp(X, *popt)
   
    return popt
figpath = './figure/'
datapath = './data/'

type1_tw_de = pd.read_csv(datapath+'type1_tw_de.txt', index_col=0)
rain = type1_tw_de[(type1_tw_de.wwflag==1)]
snow = type1_tw_de[(type1_tw_de.wwflag==2)]

params = [0, 2.5, 0.5,
           0, 8, 1]
conp, num_rain, num_snow = conp_2d(rain, snow, 'tw', 'posi_area1', params)


popt, X, Y = fit_50_contour(conp)


# Plotting

xmin=0
ymin=0
xmax = 15
ymax = 3

fig = plt.figure(figsize=(6, 5), dpi=300)
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
s1 = ax.scatter(rain.posi_area1, rain.tw, color=blue, marker='o', s=8)
s2 = ax.scatter(snow.posi_area1, snow.tw, color=red, marker='^',facecolor='none', s=12)

x = np.arange(0, xmax+0.01, 0.01)
y = monoExp(x, *popt) ## exp function
l2, = ax.plot(x, y, 'k')

lg = ax.legend([s1, s2, l1, l2], ['rain', 'snow'])
ax.set_xlim([xmin, xmax])
ax.set_ylim([xmin, ymax])
ax.set_xlabel('TwME (J/kg)')
ax.set_ylabel('Near-surface Tw ('+ chr(176) + 'C)')
ax.set_title('Type 1 Sounding\nSeparation Line Between Rain and Snow')

plt.savefig(figpath+'Figure5.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig(figpath+'Figure5', dpi=300, bbox_inches='tight')

