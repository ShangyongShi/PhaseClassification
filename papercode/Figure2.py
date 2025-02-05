#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:39:51 2024

@author: ssynj
"""


'''
Figure 3
bias estimated by t and probsnow method


'''
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./') 

import matplotlib.pyplot as plt
from function.plot_basemap import *
import cartopy.crs as ccrs
import matplotlib.colors as colors

bias = pd.read_csv('../data/Figure2/snow_fraction_bias_temp_and_probsnow.txt', dtype={'ID':str}, index_col=0)

# set projection
leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 85)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15
proj = ccrs.PlateCarree(central_longitude=0)
#%%
# set colormap
rgb = pd.read_csv( '../data/diff_16colors.txt', 
                  delim_whitespace=True, header=None).values/255
rgb = rgb[1:-1] #14 colors
cmap = colors.LinearSegmentedColormap.from_list('mymap', rgb[1:-1], len(rgb)-2)
cmap.set_under(rgb[0])
cmap.set_over(rgb[-1])

# set boundaries
bounds = [-6, -5,-4, -3, -2, -1, -0.1, 0, 0.1, 2, 4, 8, 12, 20] # in %
bounds=[-16, -8, -4, -2, -1,-0.5, 0, 0.5,1, 2, 4, 8,  16] 
boundlabels=bounds
norms = colors.BoundaryNorm(boundaries=bounds, ncolors=len(rgb)-2)
#%%
# plot
fig = plt.figure(figsize=(12/2.54, 16/2.54), dpi=1200)  
ax1 = fig.add_axes([0.10, 0.71, 0.7, 0.255], projection=proj)
ax2 = fig.add_axes([0.1, 0.38, 0.7, 0.255], projection=proj)
ax3 = fig.add_axes([0.1, 0.05, 0.7, 0.255], projection=proj)

# set properties and plot 4 subplots by loop
axs = [ax1, ax2, ax3]
titles = ['T-threshold', 'Tw-threshold' , 'Probsnow']
subtitles = ['(a)', '(b)', '(c)']
cols = ['t', 'tw', 'probsnow_tw']
for ax, title, subtitle, col in zip(axs, titles, subtitles, cols):
    plot_basemap(ax, img_extent, dlon, dlat)
    sct = ax.scatter(bias['lon'], bias['lat'], c=bias[col]*100, 
                     s=14,
                     cmap=cmap,
                     norm=norms)
    ax.set_title(title, fontsize=12)
    ax.set_title(subtitle, loc='left', fontsize=12)
    
# plot colorbar
position1 = fig.add_axes([0.85, 0.2, 0.02, 0.610])
cb = plt.colorbar(sct, cax=position1,
                  orientation='vertical', fraction=.1, extend='both')
cb.ax.tick_params(labelsize=10)
cb.set_ticks(bounds)
cb.set_ticklabels(boundlabels)
cb.set_label('Bias in Conditional Probability of Snow (%)')


# adjust lon lat lables
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.text(-166, 28, r'$\overline{bias}$'+'=%.2f'% (bias.t.mean()*100)+'%')
ax2.text(-166, 28, r'$\overline{bias}$'+'=%.2f'% (bias.tw.mean()*100)+'%')
ax3.text(-166, 28, r'$\overline{bias}$'+'=%.2f'% (bias.probsnow_tw.mean()*100)+'%')

plt.rcParams['text.usetex'] = False

# save figure
fig.savefig('../figure/'+'Figure2', dpi=1200, bbox_inches='tight')
# fig.savefig('../figure/'+'Figure2.eps', format='eps',  bbox_iches='tight')
fig.savefig('../figure/'+'Figure2.pdf', format='pdf')


