#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:21:09 2023

@author: sshi
"""

def ConpExp(x, m, t):
    '''exponential function crossing (0, 1)'''
    return m * np.exp(-t * x) + 1-m 
from sklearn.metrics import r2_score

def conp_PA(df):
    df_rain = df[df.wwflag==1]
    df_snow = df[df.wwflag==2]

    pre_rain = df_rain.loc[:, 'PA']
    pre_snow = df_snow.loc[:, 'PA']

    binsize=2
    num_rain = pd.DataFrame(data=0, index=[0], columns=np.arange(binsize/2, 200, binsize))
    num_snow = pd.DataFrame(data=0, index=[0], columns=np.arange(binsize/2, 200, binsize))
    for col in num_rain.columns:
        num_rain.loc[0, col] += ((pre_rain>col-binsize/2) & (pre_rain<=col+binsize/2)).sum()
        num_snow.loc[0, col] += ((pre_snow>col-binsize/2) & (pre_snow<=col+binsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)

    xs, ys = conp.columns.values, conp.loc[0]
    popt, pcov = opt.curve_fit(ConpExp, xs, ys)
    y_pred = ConpExp(xs, *popt)
    r2_score(ys, y_pred)

    x50 = -np.log((0.5+popt[0]-1)/popt[0])/popt[1]

    return conp, x50, popt 



import pandas as pd
import numpy as np
from scipy import optimize as opt
from skimage import measure
def conp_1d(rain, snow, var, xmin, xmax, binsize):
    '''
    example: from -10 to 10, binsize=1
    columns: -9.5 to 9.5
    
    '''
    columns = np.arange(xmin, xmax, binsize)+binsize/2
    num_rain = pd.DataFrame(data=0, index=[0], columns=columns)
    num_snow = pd.DataFrame(data=0, index=[0], columns=columns)
    for col in num_rain.columns:
        num_rain.loc[0, col] += ((rain[var]>col-binsize/2) & (rain[var]<=col+binsize/2)).sum()
        num_snow.loc[0, col] += ((snow[var]>col-binsize/2) & (snow[var]<=col+binsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)
    return conp, num_rain, num_snow

def conp_2d(rain, snow, xvar, yvar, params):
    '''
    on the plot, column is x axis, row is y axis
    
    Input:
        rain
        snow
        xvar, yvar: str
        params:  xmin, xmax, xbinsize, ymin, ymax, ybinsize
    Output:
        conp, num_rain, num_snow
    '''
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
    
    # column is x axis, row is y axis on the plot
    num_rain = pd.DataFrame(data=0, 
                            index=np.arange(ymin, ymax, ybinsize)+ybinsize/2, 
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    num_snow = pd.DataFrame(data=0, 
                            index=np.arange(ymin, ymax, ybinsize)+ybinsize/2, 
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    for row in num_rain.index:
        for col in num_rain.columns:
            num_rain.loc[row, col] += ((rain[yvar]>row-ybinsize/2) & 
                                       (rain[yvar]<=row+ybinsize/2) & 
                                       (rain[xvar]>col-xbinsize/2) & 
                                       (rain[xvar]<=col+xbinsize/2)).sum()
            num_snow.loc[row, col] += ((snow[yvar]>row-ybinsize/2) & 
                                       (snow[yvar]<=row+ybinsize/2) &
                                       (snow[xvar]>col-xbinsize/2) & 
                                       (snow[xvar]<=col+xbinsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)
    return conp, num_rain, num_snow

def fit_tanh(xs, conp):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(tanh, xs, conp)
    return popt

def tanh(x, a, b, c, d):
    import sympy as sp
    return a*(np.tanh(b*(x-c)) - d)

def cal_t50(xs, conp):
    '''
    fit tanh function to the conditional probability
    and solve the temperature at 50% probability
    
    Input: 
        xs: temperature bins
        conp: probability at bin
    '''
    import sympy as sp
    from sympy import solve, symbols
    
    # fit tanh function
    popt = fit_tanh(xs, conp)
    
    x = symbols('x')
    
    a, b, c, d = popt
    tw50 = solve(sp.tanh(x) - 0.5/a - d, x)
    tw50 = tw50/b+c
    tw50 = tw50[0]
    return tw50

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
    Y[0] = 1.28
    return X, Y



def fit_with_exp(X, Y):
    # Fit the contour with exp function
    from sklearn.metrics import r2_score
    popt, pcov = opt.curve_fit(monoExp, X, Y)
    y_pred = monoExp(X, *popt)
   
    return popt