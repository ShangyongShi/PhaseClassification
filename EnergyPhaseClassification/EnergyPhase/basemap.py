#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:44:25 2025

@author: ssynj
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
proj = ccrs.PlateCarree(central_longitude=0)

def figure_basemap(figsize=(6.5, 4), dpi=300, img_extent=[-180, 180, -90, 90], 
                   dlon=60, dlat=30, proj= ccrs.PlateCarree(central_longitude=0)):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.13, 0.15, 0.7, 0.7], projection=proj)
    plot_basemap(ax, img_extent, dlon, dlat)
    return fig, ax

def plot_basemap(ax, img_extent, dlon, dlat):
    ''' 
    basic settings of the basemap. No output. Would work on the axis.
    
    Input:
        ax1: figure axis.
        img_extent: [leftlon, rightlon, lowerlat, upperlat]
        dlon, dlat: intervals of longitude/latitude
        
    Plan to add args:
    args: 
        gridlines
        state_border
        
        
    '''
    leftlon, rightlon, lowerlat, upperlat = img_extent
    proj = ccrs.PlateCarree(central_longitude=0)
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.2)
    
    lon_formatter = cticker.LongitudeFormatter(zero_direction_label=True)
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    xticks = np.arange(leftlon//5*5, rightlon//5*5+1, dlon)
    yticks = np.arange(lowerlat//5*5, upperlat//5*5+1, dlat)
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='0.2',alpha=0.8, lw=0.2)
    ax.add_feature(cfeature.NaturalEarthFeature(
                    'cultural', 'admin_1_states_provinces_lines', '50m',
                    edgecolor='0.2',alpha=0.95, facecolor='none', lw=0.2)
                  )
    # set gridlines
    gl = ax.gridlines(crs=proj, draw_labels=False,
            linewidth=0.3, linestyle=(0, (5, 10)), color=(0.2,0.2,0.2), alpha=0.8)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    return ax