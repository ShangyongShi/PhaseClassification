# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:12:37 2023

@author: ssynj
"""
import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

from function.plot_basemap import *

figpath = './figure/'



# Read data
bias = pd.read_csv('./data/snow_fraction_biasemp_and_probsnow.txt', index_col=0)

# Read station location information
stations = pd.read_csv('./data/NCEP_IGRA_collocated_stations.txt', delim_whitespace=True)
stations.set_index('NCEP_ID', inplace=True)
# lat and lon of the stations in the data
bias_loc = stations.loc[bias.index]
bias_loc = bias_loc[~bias_loc.index.duplicated()]

# self defined colormap
rgb = pd.read_csv('./data/diff_14colors.txt', 
                  delim_whitespace=True, header=None).values/255
cmap = mpl.colors.LinearSegmentedColormap.from_list(
           'mymap', rgb[1:-1], len(rgb)-2)
cmap.set_under(rgb[0])
cmap.set_over(rgb[-1])

# define desired intervals
bounds = [-0.06, -0.05,-0.04, -0.03, -0.02, -0.01, -0.001,
          0, 
          0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
boundlabels=bounds
norms = colors.BoundaryNorm(boundaries=bounds, ncolors=len(rgb)-2)


######## plot
fig = plt.figure(figsize=(7, 5), dpi=300)  
proj = ccrs.PlateCarree(central_longitude=0)
ax1 = fig.add_axes([0.10, 0.48, 0.35, 0.33], projection=proj)
ax2 = fig.add_axes([0.52, 0.48, 0.35, 0.33], projection=proj)
ax3 = fig.add_axes([0.10, 0.12, 0.35, 0.33], projection=proj)
ax4 = fig.add_axes([0.52, 0.12, 0.35, 0.33], projection=proj)

position1 = fig.add_axes([0.93, 0.16, 0.01, 0.620])

leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 90)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15

plot_basemap(ax1, img_extent, dlon, dlat)
ax1.set_xticklabels([])
sct1= ax1.scatter(bias_loc['LON'], bias_loc['LAT'],
                  c=bias['t'],s=20,
                  cmap=cmap,
                 norm=norms)
ax1.set_title('T-threshold', fontsize=12)
ax1.set_title('(a)', loc='left', fontsize=12)
cb = plt.colorbar(sct1, cax=position1 ,orientation='vertical', fraction=.1, extend='both')
cb.ax.tick_params(labelsize=10)
cb.set_ticks(bounds)
cb.set_ticklabels(boundlabels)
cb.set_label('Bias in Snow Fraction')

plot_basemap(ax2, img_extent, dlon, dlat)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
sct2= ax2.scatter(bias_loc['LON'], bias_loc['LAT'],
                  c=bias['tw'],s=20,
                  cmap=cmap,
                  norm=norms)
ax2.set_title('Tw-threshold', fontsize=12)
ax2.set_title('(b)', loc='left', fontsize=12)

plot_basemap(ax3, img_extent, dlon, dlat)
sct3= ax3.scatter(bias_loc['LON'], bias_loc['LAT'],
                  c=bias['probsnow_t'], s=20,
                  cmap=cmap,
                  norm=norms)
ax3.set_title('TProbsnow', fontsize=12)
ax3.set_title('(c)', loc='left', fontsize=12)

plot_basemap(ax4, img_extent, dlon, dlat)
ax4.set_yticklabels([])
sct4= ax4.scatter(bias_loc['LON'], bias_loc['LAT'],
                  c=bias['probsnow_tw'],s=20,
                  cmap=cmap,
                 norm=norms)
ax4.set_title('TwProbsnow', fontsize=12)
ax4.set_title('(d)', loc='left', fontsize=12)

fig.savefig(figpath+'Figure1', dpi=300, bbox_inches='tight')
fig.savefig(figpath+'Figure1.eps', format='eps', dpi=300, bbox_iches='tight')