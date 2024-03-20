#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:08:30 2024
Figure 12

@author: ssynj
"""



import pandas as pd
import numpy as np
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
#%%    
file1 = '../data/5by5/5by5_avg_sf_prob_2008.txt'
file2 = '../data/5by5/5by5_avg_sf_energy_2008.txt'
energydat = np.loadtxt(file1, delimiter=',')
probdat = np.loadtxt(file2, delimiter=',')

lat = np.arange(-90, 90, 5)
lon = np.arange(-180, 180, 5)
df_prob_global = pd.DataFrame(data=probdat, index=lat, columns=lon)
df_energy_global = pd.DataFrame(data=energydat, index=lat, columns=lon)
#%%
df_prob5 = df_prob_global.loc[25:90, -170:-50]*24
df_energy5 = df_energy_global.loc[25:90, -170:-50]*24
# =============================================================================
# this is when loading 1 by 1 files
# # sum into 5 by 5
# df_prob5 = pd.DataFrame(data=0, index=np.arange(25, 90, 5), columns = np.arange(-170, -50, 5))
# df_energy5 = pd.DataFrame(data=0, index=np.arange(25, 90, 5), columns = np.arange(-170, -50, 5))
# for idx in range(0, 65, 5):
#     for icol in range(0, 120, 5):
#         df_prob5.iloc[int(idx/5), int(icol/5)] = df_prob.iloc[idx:idx+4, icol:icol+4].sum().sum()
#         df_energy5.iloc[int(idx/5), int(icol/5)] = df_energy.iloc[idx:idx+4, icol:icol+4].sum().sum()
# =============================================================================

# ocean mask
ocean = pd.DataFrame(data=False, 
                     index=np.arange(25, 90, 5), 
                     columns=np.arange(-170, -50, 5))
ocean.loc[55, [-155, -150, -145, -55]] = True
ocean.loc[50, np.arange(-170, -130, 5)] = True
ocean.loc[45, np.arange(-170, -125, 5)] = True
ocean.loc[40, np.arange(-170, -125, 5)] = True
ocean.loc[35, np.arange(-170, -125, 5)] = True
ocean.loc[30, np.arange(-170, -120, 5)] = True
ocean.loc[25, np.arange(-170, -115, 5)] = True
ocean.loc[40, np.arange(-60, -50, 5)] = True
ocean.loc[35, np.arange(-75, -50, 5)] = True
ocean.loc[30, np.arange(-75, -50, 5)] = True
ocean.loc[25, np.arange(-80, -50, 5)] = True
ocean.loc[25, np.arange(-95, -80, 5)] = True


diff = (df_energy5-df_prob5)
per = ((df_energy5-df_prob5)/df_prob5)

df_energy5 = df_energy5.mask(ocean)
df_prob5 = df_prob5.mask(ocean)
diff = diff.mask(ocean).values
per = per.mask(ocean).values



# set colormap
wbgyr = pd.read_csv('../data/whiterainbow.txt',
                    header=None, delim_whitespace=True)
wbgyr /= 255
wbgyr = wbgyr.values
cmap = colors.LinearSegmentedColormap.from_list(
       'mymap', wbgyr, len(wbgyr))
#  difference colormap
dfcmap = pd.read_csv('../data/darkbluedarkred.txt', header=None).values/255
br = colors.ListedColormap(dfcmap[1:-1])
br.set_under(dfcmap[0, :])
br.set_over(dfcmap[-1, :])
#%%
# set projection
leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 85)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15

figpath = '../figure/'

fig = plt.figure(figsize=(6, 8), dpi=300)  
ax1  = fig.add_axes([0.12, 0.65, 0.65, 0.215], projection=proj)
ax2  = fig.add_axes([0.12, 0.35, 0.65, 0.215], projection=proj)
ax3  = fig.add_axes([0.12, 0.05, 0.65, 0.215], projection=proj)

position1 = fig.add_axes([0.8, 0.65, 0.01, 0.215])
position2 = fig.add_axes([0.8, 0.35, 0.01, 0.215])
position3 = fig.add_axes([0.8, 0.05, 0.01, 0.215])

lon = np.arange(-170, -50+1, 5)
lat = np.arange(25, 89, 5)
p1 = ax1.pcolor(lon, lat, df_energy5.values, cmap=cmap, vmin=0, vmax=1)
plot_basemap(ax1, img_extent, dlon, dlat)

p2 = ax2.pcolor(lon, lat, diff, cmap=br, vmin=-0.1, vmax=0.1)
plot_basemap(ax2, img_extent, dlon, dlat)
             
p3 = ax3.pcolor(lon, lat, per*100, cmap=br, vmin=-15, vmax=15)
plot_basemap(ax3, img_extent, dlon, dlat)

cb1 = plt.colorbar(p1, cax=position1, orientation='vertical', extend='max')
cb2 = plt.colorbar(p2, cax=position2, orientation='vertical', extend='both')
cb3 = plt.colorbar(p3, cax=position3, orientation='vertical', extend='both')
cb1.ax.set_ylabel('mm/day')
cb2.ax.set_ylabel('mm/day')
cb3.ax.set_ylabel('Percentage of change')
cb3.ax.set_title('%')

ax1.set_title('Mean Snowfall Rate (Energy)')
ax2.set_title('Energy - TwProbsnow')
ax3.set_title('Relative Difference to TwProbsnow')

ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')

fig.savefig(figpath+'Figure12', dpi=1200, bbox_inches='tight')
fig.savefig(figpath+'Figure12.pdf', format='pdf', bbox_inches='tight')



#%%
# -----------
datapath = '../data/5by5/'
# =============================================================================
# def regrid_5by5(n0):
#     # sum into 5 by 5
#     df5 = pd.DataFrame(data=0, index=np.arange(25, 90, 5), columns = np.arange(-170, -50, 5))
#     for idx in range(0, 65, 5):
#         for icol in range(0, 120, 5):
#             df5.iloc[int(idx/5), int(icol/5)] = n0.iloc[idx:idx+4, icol:icol+4].sum().sum()
#     return df5
# =============================================================================


def readn(file):
    lat = np.arange(-90, 90, 5)
    lon = np.arange(-180, 180, 5)
    dat = np.loadtxt(file, delimiter=',')
    df = pd.DataFrame(data=dat, index=lat, columns=lon)
    df = df.loc[25:90, -170:-50]
    return df



lon = np.arange(-170, -50+1, 5)
lat = np.arange(25, 89, 5)
# df_prob_global = pd.DataFrame(data=probdat, index=lat, columns=lon)
# df_energy_global = pd.DataFrame(data=energydat, index=lat, columns=lon)

n0 = readn(datapath+'5by5_ntype0_2008.txt')
n1 = readn(datapath+'5by5_ntype1_2008.txt')
n2 = readn(datapath+'5by5_ntype2_2008.txt')
# n0 = regrid_5by5(n0)
# n1 = regrid_5by5(n1)
# n2 = regrid_5by5(n2)

frac1 = n1/(n1+n2)
frac2 = n2/(n1+n2)

frac2 = frac2.mask(ocean)


cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (1,1,1, 1.0)
# create the new map
cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist[:-20], cmap.N-20)
cmap.set_over(cmaplist[-1])

fig = plt.figure(figsize=(7, 6), dpi=1000)  
ax1  = fig.add_axes([0.15, 0.55, 0.65, 0.3], projection=proj)
position1 = fig.add_axes([0.78, 0.55, 0.01, 0.3])


bounds = [0,0.001, 0.01,0.02, 0.04,0.1, 0.2, 0.3]
norms = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

p1 = ax1.pcolor(lon, lat, frac2.values, cmap=cmap, norm=norms)
plot_basemap(ax1, img_extent, dlon, dlat)      

cb1 = plt.colorbar(p1, cax=position1, orientation='vertical', extend='max')

cb1.ax.set_ylabel('fraction (excluding Type 0)')
ax1.set_title('Fraction of Type 2 Soundings in Snow')
fig.savefig(figpath+'era5_frac_type2', dpi=1000, bbox_inches='tight')
