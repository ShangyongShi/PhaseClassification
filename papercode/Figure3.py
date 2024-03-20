#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:19:50 2024

@author: ssynj
"""

import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
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
    yticks = np.arange(lowerlat//5*5, upperlat//5*5+5, dlat)
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
'''
Figure 2
sounding percentage
'''
def drawPieMarker(ax, xs, ys, ratios, sizes, ncolors):
    '''
    draw a pie marker at a given location
    https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    
    xs, ys: arrays of xs and ys
    
    '''
    # assert sum(ratios) <= 1, 'sum of ratios needs to be <= 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(ncolors, ratios):
        this = 2 * np.pi * ratio + previous # angle
        x  = [0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0] # polygon
        y  = [0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        
        #for wedges that never reach the min or max x or y, you need a rescaling
        sizes_scaled = np.abs(xy).max()**2*np.array(sizes)  
        
        markers.append({'marker':xy, 's':sizes_scaled, 'facecolor':color}) 

    # scatter each of the pie pieces to create pies
    for marker in markers:
        sct = ax.scatter(xs, ys, **marker)
    return sct




def count_num_sounding(df_t10):
    '''
    Output: 
        num_sounding_t10, with columns 'total', 'type0', 'type1', 'type2' 
            and the station ID as indexes
            
            total: number of precipitation events
            type0: record the number of soundings excluding all warm or all cold ones.
            type1 is one warm layer (cat2)
            type2 is melting layer + refreezing layer
        
    '''
    type0_t10 = df_t10[df_t10['type_tw']==0]
    type1_t10 = df_t10[df_t10['type_tw']==1]
    type2_t10 = df_t10[df_t10['type_tw']==2]
        
    ID_t10 = df_t10.ID.unique()
    num_sounding_t10 = pd.DataFrame(data=0, index=ID_t10, 
                                    columns=['total', 'type0', 'type1', 'type2'])
    for ID in ID_t10:
        num_sounding_t10.loc[ID, 'total'] = len(df_t10[df_t10.ID==ID]) # -len(cat0_t10[cat0_t10.ID==ID])-len(cat1_t10[cat1_t10.ID==ID])
        num_sounding_t10.loc[ID, 'type0'] = len(type0_t10[type0_t10.ID==ID])
        num_sounding_t10.loc[ID, 'type1'] = len(type1_t10[type1_t10.ID==ID])
        num_sounding_t10.loc[ID, 'type2'] = len(type2_t10[type2_t10.ID==ID])
    return num_sounding_t10

def cal_sounding_percentage(num_sounding_t10):
    # percentage in all soundings
    ratio_pre_t10 = pd.DataFrame(index=num_sounding_t10.index, 
                                 columns=['type0', 'type1', 'type2'])
    ratio_pre_t10['type0'] = num_sounding_t10['type0']/num_sounding_t10['total']
    ratio_pre_t10['type1'] = num_sounding_t10['type1']/num_sounding_t10['total']
    ratio_pre_t10['type2'] = num_sounding_t10['type2']/num_sounding_t10['total']
    return ratio_pre_t10

def cal_sounding_percentage_excluding_type0(num_sounding_t10):
    ratio_pre_t10 = pd.DataFrame(index=num_sounding_t10.index, 
                                 columns=['type0', 'type1', 'type2'])
    ratio_pre_t10['type0'] = np.nan
    total = num_sounding_t10['type1'] + num_sounding_t10['type2'] 
    ratio_pre_t10['type1'] = num_sounding_t10['type1']/total
    ratio_pre_t10['type2'] = num_sounding_t10['type2']/total
    return ratio_pre_t10

figpath = '../figure/'

stations = pd.read_csv('../data/NCEP_IGRA_collocated_stations_cleaned.txt')
stations.set_index('NCEP_ID', inplace=True)


path = '../data/'
final = pd.read_csv(path + 'final_all_cleaned_data.txt', 
                    index_col=0, dtype={'ID':str})

df_ti10_snow = final[(final.ti>=-10) & (final.ti<=10) & (final.wwflag==2)]

num_sounding_ti10_snow = count_num_sounding(df_ti10_snow)
ratio_ti10_snow = cal_sounding_percentage(num_sounding_ti10_snow)
ratio_excluded_ti10_snow = cal_sounding_percentage_excluding_type0(num_sounding_ti10_snow)


leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 85)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15
proj = ccrs.PlateCarree(central_longitude=0) 

blue = np.array([14, 126, 191])/255
red = np.array([235, 57, 25])/255
green = np.array([77,175,74])/255
green = np.array([200,200,200])/255

# plot
fig = plt.figure(figsize=(16/2.54, 3), dpi=1200)  
ax  = fig.add_axes([0.07, 0.11, 0.75, 0.75], projection=proj)

plot_basemap(ax, img_extent, dlon, dlat)
stations_t10 = stations.loc[ratio_ti10_snow.index]

for ID in stations_t10.index:
    num = num_sounding_ti10_snow.loc[ID, 'total']
    if num<=50:
        rsize = 15
    elif num<=200:
        rsize=30
    elif num<=500:
        rsize=45
    else:
        rsize=60
        
    if ratio_ti10_snow.loc[ID, 'type0']==1:
        drawPieMarker(ax,
                      xs=stations_t10.loc[ID, 'LON'],
                      ys=stations_t10.loc[ID, 'LAT'],
                      ratios=[1],
                      sizes=[rsize],
                      ncolors=[green])
        continue
        
    if ratio_excluded_ti10_snow.loc[ID, 'type1'] == 1:
        ratios = [1]
        ncolors = [blue]
    elif ratio_excluded_ti10_snow.loc[ID, 'type2'] == 0:
        ratios = [ratio_excluded_ti10_snow.loc[ID, 'type1']]
        ncolors = [blue]
    else:
        ratios = [
                  ratio_excluded_ti10_snow.loc[ID, 'type1'], 
                  ratio_excluded_ti10_snow.loc[ID, 'type2'],
                 ]
        ncolors = [blue, red]
    drawPieMarker(ax,
                  xs=stations_t10.loc[ID, 'LON'],
                  ys=stations_t10.loc[ID, 'LAT'],
                  ratios=ratios,
                  sizes=[rsize],
                  ncolors=ncolors)

ax.set_title('Sounding Percentage in Snow Events', fontsize=12)


sct11 = drawPieMarker(ax, 30, 40, [1], [80], [blue])
sct12 = drawPieMarker(ax, 30, 40, [1], [60], [blue])
sct13 = drawPieMarker(ax, 30, 40, [1], [40], [blue])
sct14 = drawPieMarker(ax, 30, 40, [1], [20], [blue])
legend2 = plt.legend([sct11, sct12, sct13, sct14], 
                     ['>500', '200-500', '50-200', '<50'], 
                     bbox_to_anchor=[1, .45],
                     frameon=False,
                     labelspacing=0.7,
                     fontsize=10
                    )
ax.add_artist(legend2)

sct0 = ax.scatter(30, 0, facecolor=green)
sct1 = ax.scatter(30, 0, facecolor=blue)
sct2 = ax.scatter(30, 0, facecolor=red)
ax.legend( [sct0, sct1, sct2], 
           ['Type0', 'Type1','Type2'],
          bbox_to_anchor=[1.21, 1.03],
          fontsize=8, labelspacing=0.7)
           # 

ax.text(1.04, 0.5, '$N_{snow\ soundings}$',transform=ax.transAxes)

# plt.savefig('../03-Figure/20220530_scheme/'+'Figure3', dpi=300, bbox_inches='tight')
plt.savefig(figpath+'Figure3', dpi=1200, bbox_inches='tight')
plt.savefig(figpath+'Figure3.pdf',format='pdf')


