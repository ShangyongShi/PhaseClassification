#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:21:48 2023

@author: sshi
"""
import matplotlib.pyplot as plt
import os
import numpy as np


blue = np.array([14, 126, 191])/255
red = np.array([235, 57, 25])/255
 

from function.type1 import *
from function.type2 import *
def plot_conp_PA(ax, conp, x50, popt, params):
    l1, = ax.plot(conp.columns.values, conp.loc[0], 'r.-')
    xfit = np.arange(0, 150, 0.1)
    l2, = ax.plot(xfit, ConpExp(xfit, *popt), 'k-', alpha=0.8)
    ax.plot([0, 1000], [0.5, 0.5], 'k--', alpha=0.6)
    l3, = ax.plot([x50, x50], [0, 1], 'k--', alpha=0.6)
    ax.set_xticks(np.arange(0, 60, 10))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.legend([l1, l2, l3], ['Obs', 'Fit', 'Threshold={:4.1f}'.format(x50)+' J kg$^{-1}$'])
    ax.set_xlabel('Melting Energy (J/Kg)')
    ax.set_ylabel('Conditional Probability of Solid Precipitation')
    ax.set_title(params['title'])
    return

def plot_conp_ME_Tw(conp):
    fig = plt.figure(figsize=(5, 4), dpi=300)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    
    # shadings
    pc = ax.contourf(conp.index, conp.columns, conp.T, 
                     cmap='jet', 
                     origin='lower', levels=np.arange(0, 1.1, 0.1))
    # contours
    ct = ax.contour(conp.index, conp.columns, conp.T, 
                    colors='k', linewidths=0.5,
                    origin='lower', levels=np.arange(0, 1.1, 0.1))
    ax.clabel(ct, ct.levels, inline=True, fmt='%3.1f', fontsize=10)
    
    # plot the 50 contour
    popt, X, Y = fit_50_contour(conp)
    x = np.arange(1, 7., 0.1)
    y = monoExp(x, *popt)## fit function
    ax.plot(x, y, 'k')
    # ax.plot(X, Y)
    cb = plt.colorbar(pc, cax=fig.add_axes([0.9, 0.15, 0.03, 0.7]), 
                      label='Conditional Probability of Solid Precipitation')
    ax.set_xlabel('TwME (J/kg)')
    ax.set_ylabel('Near surface Tw ('+chr(176)+'C)')
    ax.set_ylim([0.25, 2.25])
    ax.set_xticks(np.arange(1, 8, 1))
    return fig, ax, popt





def plot_type2_scatter_LDA(rain, snow, PA, NA, slope, intercept):
    ''' Scatter plot and LDA '''
    fig = plt.figure(figsize=(5, 4), dpi=300)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    s1 = ax.scatter(rain[PA], rain[NA], color=blue, marker='o', facecolor='none', s=10, alpha=0.8)
    s2 = ax.scatter(snow[PA], snow[NA], color=red, marker='^',facecolor='none', s=14, alpha=0.6)
    
    
    xs = np.arange(0, 500+0.1, 0.1)
    l1, = ax.plot(xs, (slope*xs)+intercept,'--', color=[.26,.26,.26], linewidth=3) 
    
    lg1 = ax.legend([s1, s2, l1],
                    ['rain', 'snow',
                      'LDA boundary line']               
                      )
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 500])
    ax.set_xlabel('TwME (J/kg)')
    ax.set_ylabel('TwRE (J/kg)')
    ax.set_title('Type 2 Sounding\nLDA Separation Line Between Rain and Snow')
    return fig, ax

def plot_type2_scatter(rain, snow, xvar, yvar, params):
    ''' Scatter plot and LDA '''
    fig = plt.figure(figsize=(5, 4), dpi=300)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    s1 = ax.scatter(rain[xvar], rain[yvar], color=blue, marker='o', facecolor='none', s=10, alpha=0.8)
    s2 = ax.scatter(snow[xvar], snow[yvar], color=red, marker='^',facecolor='none', s=14, alpha=0.6)
    lg1 = ax.legend([s1, s2],
                    ['rain', 'snow'])
    ax.set_ylim(params['ylim'])
    ax.set_ylabel(params['ylabel'])
    ax.set_xlim(params['xlim'])
    ax.set_xlabel(params['xlabel'])
    ax.set_title(params['title'])
    return fig, ax


def plot_type2_contour(conp, params):
    # contour plot
    fig = plt.figure(figsize=(5, 4), dpi=300)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    # shadings
    pc = ax.contourf(conp.index, conp.columns, conp.T, 
                     cmap='jet', 
                     origin='lower', levels=np.arange(0, 1.1, 0.1))
    # contours
    ct = ax.contour(conp.index, conp.columns, conp.T, 
                    colors='k', linewidths=0.5,
                    origin='lower', levels=np.arange(0, 1.1, 0.1))
    ax.clabel(ct, ct.levels, inline=True, fmt='%3.1f', fontsize=10)
    cb = plt.colorbar(pc, cax=fig.add_axes([0.9, 0.15, 0.03, 0.7]), 
                      label='Conditional Probability of Snow')
    ax.set_ylim(params['ylim'])
    ax.set_ylabel(params['ylabel'])
    ax.set_xlim(params['xlim'])
    ax.set_xlabel(params['xlabel'])
    return fig, ax
