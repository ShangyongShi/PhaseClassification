# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:42:08 2023

@author: ssynj
"""
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
proj = ccrs.PlateCarree(central_longitude=0)
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
    
    xticks = np.arange(leftlon//5*5, rightlon//5*5+5+1, dlon)
    yticks = np.arange(lowerlat//5*5, upperlat//5*5+1, dlat)
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.2)
    ax.add_feature(cfeature.NaturalEarthFeature(
                   'cultural', 'admin_1_states_provinces_lines', '50m',
                   edgecolor='black', facecolor='none', lw=0.2)
                  )
    # set gridlines
    gl = ax.gridlines(crs=proj, draw_labels=False,
            linewidth=0.3, linestyle=':', color=(0.2,0.2,0.2), alpha=0.8)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)