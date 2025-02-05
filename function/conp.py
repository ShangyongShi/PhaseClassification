#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:16:29 2023

@author: sshi
"""
import numpy as np
import pandas as pd

def conp_1d(rain, snow, var, xmin, xmax, binsize):
    '''
    return conditional probability of solid precipitation based on var

    example: from -10 to 10, binsize=1
    columns: -9.5 to 9.5
    
    INput:
        rain, snow: DataFrame
        var: string, variable name
        xmin, xmax: float, desired range
        binsize: int
    
    '''
    indexs = np.arange(xmin, xmax, binsize)+binsize/2
    num_rain = pd.DataFrame(data=0, index=indexs, columns=[0])
    num_snow = pd.DataFrame(data=0, index=indexs, columns=[0])
    for col in num_rain.columns:
        num_rain.loc[0, col] += ((rain[var]>col-binsize/2) & (rain[var]<=col+binsize/2)).sum()
        num_snow.loc[0, col] += ((snow[var]>col-binsize/2) & (snow[var]<=col+binsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)
    return conp, num_rain, num_snow

def fit_tanh(xs, conp):
    '''
    fit tanh function to conditional probability
    a*(np.tanh(b*(x-c)) - d)
    
    Input:
        xs, conp: arrays with the same length
    Output:
        popt: coefficients of tanh function
    '''
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(tanh, xs, conp)
    return popt

def tanh(x, a, b, c, d):
    '''a*(np.tanh(b*(x-c)) - d)'''
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

def conp_2d(rain, snow, xvar, yvar, params):
    '''
    on the plot, column is x axis, row is y axis
    
    Input:
        rain, snow: DataFrames
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


    
def running_conp(rain, snow, xvar, yvar, params):
    '''
    calculate the running counts and probability
    the summing grids will overlap
    
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
    
    '''
    # xvar, yvar = 'tw', 'PA'
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
        # column: xvar
        # row: yvar
        # when plotting, we plot the transpose of the matrix
    # column is x axis, row is y axis on the plot
    num_rain = pd.DataFrame(data=0, 
                            index=np.round(np.arange(ymin, ymax+0.1, 0.1),decimals=1), 
                            columns=np.round(np.arange(xmin, xmax+0.1, 0.1), decimals=1))
    num_snow = pd.DataFrame(data=0, 
                            index=np.round(np.arange(ymin, ymax+0.1, 0.1),decimals=1), 
                            columns=np.round(np.arange(xmin, xmax+0.1, 0.1), decimals=1))
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
    conp = num_snow/(num_rain+num_snow)
    return conp, num_rain, num_snow


def running_mean(conp2, d):
    '''
    calculate the running mean within d*d box for conp2
    '''
    cut = np.round(d/2, decimals=1)
    newcol = np.round(np.arange(conp2.columns[0]+cut, conp2.columns[-1]-cut+0.1, 0.1), decimals=1)
    newrow = np.round(np.arange(conp2.index[0]+cut, conp2.index[-1]-cut+0.1, 0.1), decimals=1)
    conp_smooth = pd.DataFrame(data=np.nan, index=newrow, columns=newcol)

    for row in newrow:
        for col in newcol:
            tmp = conp2.loc[row-cut:row+cut, col-cut:col+cut]
            conp_smooth.loc[row, col] = tmp.sum().sum()/ (tmp.shape[0]*tmp.shape[1])
    conp_smooth[conp_smooth==0] = np.nan
    return conp_smooth 
    
