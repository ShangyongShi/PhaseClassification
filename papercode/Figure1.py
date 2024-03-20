#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:39:37 2024

@author: ssynj
"""


import pandas as pd
import numpy as np
import os
import sys
sys.path.append('C:/Files/Research/04-Temp_Threshold/nimbus/') 

import matplotlib.pyplot as plt
from function.plot_basemap import *
import cartopy.crs as ccrs
import matplotlib.colors as colors

figpath = '../figure/'

#stations = pd.read_csv('/r1/sshi/sounding_phase/data/NCEP_IGRA_collocated_stations_cleaned.txt')
stations = pd.read_csv('../data/NCEP_IGRA_collocated_stations_cleaned.txt')
stations.set_index('NCEP_ID', inplace=True)


'''
Figure 1

sounding types

'''
sounding0A = pd.read_csv('../data/sounding/sounding0A.txt')
sounding0B = pd.read_csv('../data/sounding/sounding0B.txt')
sounding1 = pd.read_csv('../data/sounding/sounding1.txt')
sounding2 = pd.read_csv('../data/sounding/sounding2.txt')
p01, t01 = sounding0A['p'].values, sounding0A['t'].values
p02, t02 = sounding0B['p'].values, sounding0B['t'].values
p1, t1 = sounding1['p'].values, sounding1['t'].values
p2, t2 = sounding2['p'].values, sounding2['t'].values

def linear(t1, p1, t2, p2):
    slope = (p2 - p1)/(t2 - t1)
    intercept = p1 - slope*t1
    return slope, intercept

def sounding_linear_functions(xs, ys):
    nlines = len(xs)-1
    slope, intercept = np.zeros(nlines), np.zeros(nlines)

    for i in range(nlines):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i+1], ys[i+1]
        slope[i], intercept[i] = linear(x1, y1, x2, y2)        
    return slope, intercept


lightblue = np.array([182, 216, 235])/255
lightred = np.array([248, 195, 185])/255

fig = plt.figure(figsize=(7, 7), dpi=1000) 
position1 = [0.10, 0.55, 0.3, 0.33]
position2 = [0.55, 0.55, 0.3, 0.33]
position3 = [0.10, 0.1, 0.3, 0.33]
position4 = [0.55, 0.1, 0.3, 0.33]

ax1 = fig.add_axes(position1)
ax2 = fig.add_axes(position2)
ax3 = fig.add_axes(position3)
ax4 = fig.add_axes(position4)

ax1.plot(t01, p01, 'k.-')
ax1.plot([0, 0], [-2000, 2000], '--', color='tab:red')
ax1.plot([-20, 26], [max(p01), max(p01)], 'k--')
ax1.set_xlabel('Temperature ('+chr(176)+'C)')
ax1.set_ylabel('Pressure (hPa)')
# ax1.grid()
ax1.set_xlim([-15, 1])
ax1.set_ylim([1020, 800])
ax1.set_yticks(np.arange(1000, 750, -50))
ax1.set_title('Type 0A')

ax2.plot(t02, p02, 'k.-')
ax2.plot([0, 0], [-2000, 2000], '--', color='tab:red')
ax2.plot([-20, 30], [max(p02), max(p02)], 'k--')
ax2.set_xlabel('Temperature ('+chr(176)+'C)')
ax2.set_ylabel('Pressure (hPa)')
ax2.set_xlim([-1, 25])
ax2.set_ylim([1020, 800])
ax2.set_yticks(np.arange(1000, 750, -50))
ax2.set_title('Type 0B')

ax3.plot(t1, p1, 'k.-')
ax3.plot([0, 0], [0, 2000], '--', color='tab:red')
ax3.plot([-20, 20], [max(p1), max(p1)], 'k--')
ax3.set_xlabel('Temperature ('+chr(176)+'C)')
ax3.set_ylabel('Pressure (hPa)')
ax3.set_xlim([-7, 7])
ax3.set_ylim([1000, 800])
ax3.set_yticks(np.arange(1000, 750, -50))
ax3.set_title('Type 1')

ax4.plot(t2, p2, 'k.-')
ax4.plot([0, 0], [0, 2000], '--', color='tab:red')
ax4.plot([-20, 20], [max(p2), max(p2)], 'k--')
ax4.set_xlabel('Temperature ('+chr(176)+'C)')
ax4.set_ylabel('Pressure (hPa)')
ax4.set_xlim([-8, 8])
ax4.set_ylim([1020, 800])
ax4.set_yticks(np.arange(1000, 750, -50))
ax4.set_title('Type 2')

ax1.set_title('(a)', loc='left', fontsize=14)
ax2.set_title('(b)', loc='left', fontsize=14)
ax3.set_title('(c)', loc='left', fontsize=14)
ax4.set_title('(d)', loc='left', fontsize=14)

# fill the patches with blue or red to indicate freezing or melting layer
def slope(tt1, tt2, pp1, pp2):
    return (pp2-pp1)/(tt2-tt1)
def intercept(tt1, tt2, pp1, pp2):
    return pp1 - tt1* (pp2-pp1)/(tt2-tt1)

# (a) fill
t800 = (800-intercept(t01[4], t01[5], p01[4], p01[5]) )/slope(t01[4], t01[5], p01[4], p01[5])
x = [t800] + np.flipud(t01[0:5]).tolist() +[0]
lower_bound = [800] + np.flipud(p01[0:5]).tolist() + [p01[0]]
upper_bound = [800]*7
ax1.fill_between(x, lower_bound, upper_bound, 
                 color=lightblue, edgecolor='none', interpolate=True)
 #(b) fill
t800 = (800-intercept(t02[4], t02[5], p02[4], p02[5]) )/slope(t02[4], t02[5], p02[4], p02[5])
xx1 = [0] + [t800] + np.flipud(t02[0:5]).tolist() 
yy1 = [800, 800] + np.flipud(p02[0:5]).tolist()
yy2 = [p02[0]]*7
ax2.fill_between(xx1, yy1, yy2,
                 color=lightred, interpolate=True)

# (c) fill
surface=[p1[0], p1[0]]
warm = [ intercept(t1[0], t1[1], p1[0], p1[1]) , p1[0]]
ax3.fill_between([0, t1[0]], surface, warm, 
                 color=lightred, interpolate=True)

# (d) blue fill
x1 = np.arange(t2[2], t2[1], 0.01)
x0 = np.arange(t2[1], t2[0], 0.01)
x00 = np.arange(t2[0], 0.01, 0.01)
x2 = np.arange(t2[2], t2[3], 0.01)
x3 = np.arange(t2[3], 0.01, 0.01)

x = np.concatenate([x2, x3])

slope, intercept = sounding_linear_functions(t2[0:5], p2[0:5])
lower_bound = np.concatenate((
                    x1*slope[1] + intercept[1],
                    x0*slope[0] + intercept[0],    
                        ))
upper_bound = np.concatenate((
                    x2*slope[2]+intercept[2],
                    x3*slope[3]+intercept[3]
                            ))
lower_bound = np.concatenate((lower_bound, 
                              np.array([p2[0]]*(len(upper_bound)-len(lower_bound)))
                             ))
ax4.fill_between(x, lower_bound, upper_bound, 
                 color=lightblue, edgecolor='none', interpolate=True)


# (d) red fill
x1 = np.arange(0,      t2[7], 0.01)
x2 = np.arange(t2[7], t2[6], 0.01)
x3 = np.arange(t2[6], t2[5], 0.01)
x4 = np.arange(t2[5], t2[4]+0.01, 0.01)
x = np.concatenate([x1, x2, x3, x4])

slope, intercept = sounding_linear_functions(t2[3:9], p2[3:9])
lower_bound = slope[0]*x + intercept[0]
upper_bound = np.concatenate((
                    x1*slope[-1]+intercept[-1],
                    x2*slope[-2]+intercept[-2],
                    x3*slope[-3]+intercept[-3],
                    x4*slope[-4]+intercept[-4]))
ax4.fill_between(x, upper_bound, lower_bound, 
                  color=lightred, edgecolor='none', interpolate=True)

ax1.text(0.5, 0.55, 'All below\nfreezing', transform=ax1.transAxes)
ax2.text(0.2, 0.55, 'All above\nfreezing', transform=ax2.transAxes)
ax3.annotate("Melting layer", xy=(2, 960), xytext=(0.5, 900),
             arrowprops=dict(arrowstyle="->"))
ax4.annotate("Melting layer", xy=(1, 880), xytext=(0.33, 825),
             arrowprops=dict(arrowstyle="->"))
ax4.annotate("Refreezing\nlayer", xy=(-2.5, 970), xytext=(-6.4, 900),
             arrowprops=dict(arrowstyle="->"))

plt.savefig(figpath+'Figure1', bbox_inches='tight',dpi=1200)
#plt.savefig(figpath+'Figure1.eps', format='eps', bbox_inches='tight',dpi=300)
plt.savefig(figpath+'Figure1.pdf', format='pdf', bbox_inches='tight')

