# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:04:04 2023

@author: ssynj
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from function.plot_basemap import plot_basemap

blue = np.array([14, 126, 191])/255
red = np.array([235, 57, 25])/255
gray = np.array([0.8, 0.8, 0.8])
green = np.array([77,175,74])/255

def drawPieMarker(ax, xs, ys, ratios, sizes, colors):
    '''
    draw a pie marker at a given location
    https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    
    xs, ys: arrays of xs and ys
    
    '''
    assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x  = [0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0]
        y  = [0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker':xy, 's':np.array(sizes), 'facecolor':color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        sct = ax.scatter(xs, ys, **marker)
    return sct

def categorize(df):
    '''
    determine sounding type based on number of positive and negative
    areas and freezing level
    
    return: type0, type1, type2, others
    '''
    nPA = (~df.loc[:, 'posi_area1':'posi_area3'].isna()).sum(axis=1)
    nNA = (~df.loc[:, 'nega_area1':'nega_area3'].isna()).sum(axis=1)
    nFL = (~df.loc[:, 'freezing_level1':'freezing_level3'].isna()).sum(axis=1)
    
    type0 = df.loc[
                   ((nPA==0) & (nNA==0) & (nFL==0)) |
                   ((nPA==0) & (nNA==1) & (nFL==0)) |
                   ((nPA==1) & (nNA==0) & (nFL==0)) 
                  ]
    type1 = df.loc[
                   ((nPA==1) & (nNA==0) & (nFL==1)) 
                  ]  
    type2 = df.loc[
                   ((nPA==1) & (nNA==1) & (nFL==1)) 
                  ] 
    others= df.loc[
                   ((nPA==1) & (nNA==2) & (nFL==1)) |
                   ((nPA==2) & (nNA==0) & (nFL==1)) |
                   ((nPA==2) & (nNA==1) & (nFL==1)) |
                   ((nPA==2) & (nNA==1) & (nFL==2)) |
                   ((nPA==2) & (nNA==2) & (nFL==2)) 
                  ]
    # bad data: 1 1 0
    return type0, type1, type2, others


def rename_df_columns(df, tstr):
    '''rename columns of df
    if tstr is 't', then rename those with subscripts _t
    if tstr is 'tw', then remove the subscripts _tw'''
    if tstr=='t':
        columns_mapper={'posi_area1_t':'posi_area1', 
                        'posi_area2_t':'posi_area2', 
                        'posi_area3_t':'posi_area3', 
                        'nega_area1_t':'nega_area1', 
                        'nega_area2_t':'nega_area2', 
                        'nega_area3_t':'nega_area3',
                        'freezing_level1_t':'freezing_level1', 
                        'freezing_level2_t':'freezing_level2', 
                        'freezing_level3_t':'freezing_level3'}
    elif tstr=='tw':
        columns_mapper={'posi_area1_tw':'posi_area1', 
                    'posi_area2_tw':'posi_area2', 
                    'posi_area3_tw':'posi_area3', 
                    'nega_area1_tw':'nega_area1', 
                    'nega_area2_tw':'nega_area2', 
                    'nega_area3_tw':'nega_area3',
                    'freezing_level1_tw':'freezing_level1', 
                    'freezing_level2_tw':'freezing_level2', 
                    'freezing_level3_tw':'freezing_level3'}
    df1 = df.copy()
    df1.rename(columns=columns_mapper, inplace=True)
    return df1

datapath = './data/'
figpath = './figure/'

final_data = pd.read_csv(datapath+'final_all_cleaned_data.txt', index_col=0)
df_t = rename_df_columns(final_data, 't')
df_tw = rename_df_columns(final_data, 'tw')

df_t10 = df_t[(df_t.t>=-10) & (df_t.t<=10)]
df_tw10 = df_tw[(df_tw.tw>=-10) & (df_tw.tw<=10)]
df_t10_snow = df_t10[df_t10.wwflag==2]
df_tw10_snow = df_tw10[df_tw10.wwflag==2]

def count_num_sounding(df_t10):
    '''
    Output: 
        num_sounding_t10, with columns 'total', 'type0', 'type1', 'type2' and 'othhers'
            and the station ID as indexes
            total: number of precipitation events
            type0: record the number of soundings excluding all warm or all cold ones.
            type1 is one warm layer (cat2)
            type2 is melting layer + refreezing layer
        
    '''
    type0_t10, type1_t10, type2_t10, others_t10 = categorize(df_t10)
    
    ID_t10 = df_t10.ID.unique()
    num_sounding_t10 = pd.DataFrame(data=0, index=ID_t10, columns=['total', 'type0', 'type1', 'type2', 'others'])
    for ID in ID_t10:
        num_sounding_t10.loc[ID, 'total'] = len(df_t10[df_t10.ID==ID]) # -len(cat0_t10[cat0_t10.ID==ID])-len(cat1_t10[cat1_t10.ID==ID])
        num_sounding_t10.loc[ID, 'type0'] = len(type0_t10[type0_t10.ID==ID])
        num_sounding_t10.loc[ID, 'type1'] = len(type1_t10[type1_t10.ID==ID])
        num_sounding_t10.loc[ID, 'type2'] = len(type2_t10[type2_t10.ID==ID])
        num_sounding_t10.loc[ID, 'others'] = len(others_t10[others_t10.ID==ID])
    return num_sounding_t10

num_sounding_t10 = count_num_sounding(df_t10)
num_sounding_t10_snow = count_num_sounding(df_t10_snow)

num_sounding_tw10 = count_num_sounding(df_tw10)
num_sounding_tw10_snow = count_num_sounding(df_tw10_snow)

def cal_sounding_percentage(num_sounding_t10):
    # percentage in all soundings
    ratio_pre_t10 = pd.DataFrame(index=num_sounding_t10.index, columns=['type0', 'type1', 'type2', 'others'])
    ratio_pre_t10['type0'] = num_sounding_t10['type0']/num_sounding_t10['total']
    ratio_pre_t10['type1'] = num_sounding_t10['type1']/num_sounding_t10['total']
    ratio_pre_t10['type2'] = num_sounding_t10['type2']/num_sounding_t10['total']
    ratio_pre_t10['others'] = num_sounding_t10['others']/num_sounding_t10['total']
    return ratio_pre_t10

def cal_sounding_percentage_excluding_type0(num_sounding_t10):
    ratio_pre_t10 = pd.DataFrame(index=num_sounding_t10.index, columns=['type0', 'type1', 'type2', 'others'])
    ratio_pre_t10['type0'] = np.nan
    total = num_sounding_t10['type1'] + num_sounding_t10['type2'] + num_sounding_t10['others']
    ratio_pre_t10['type1'] = num_sounding_t10['type1']/total
    ratio_pre_t10['type2'] = num_sounding_t10['type2']/total
    ratio_pre_t10['others'] = num_sounding_t10['others']/total
    return ratio_pre_t10

ratio_t10 = cal_sounding_percentage(num_sounding_t10)
ratio_tw10 = cal_sounding_percentage(num_sounding_tw10)
ratio_t10_snow = cal_sounding_percentage(num_sounding_t10_snow)
ratio_tw10_snow = cal_sounding_percentage(num_sounding_tw10_snow)

ratio_excluded_t10 = cal_sounding_percentage_excluding_type0(num_sounding_t10)
ratio_excluded_tw10 = cal_sounding_percentage_excluding_type0(num_sounding_tw10)
ratio_excluded_t10_snow = cal_sounding_percentage_excluding_type0(num_sounding_t10_snow)
ratio_excluded_tw10_snow = cal_sounding_percentage_excluding_type0(num_sounding_tw10_snow)

'''
Plotting part
'''
proj = ccrs.PlateCarree(central_longitude=0)
leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 90)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15

fig = plt.figure(figsize=(7.5, 6), dpi=300)  
ax1  = fig.add_axes([0.15, 0.55, 0.65, 0.3], projection=proj)
plot_basemap(ax1, img_extent, dlon, dlat)

stations = pd.read_csv(datapath+'NCEP_IGRA_collocated_stations.txt', delim_whitespace=True)
stations.set_index('NCEP_ID', inplace=True)

stations_t10 = stations.loc[ratio_tw10.index]
for ID in stations_t10.index:
    num = num_sounding_tw10.loc[ID, 'total']
    if num<=50:
        rsize = 20
    elif num<=200:
        rsize=40
    elif num<=500:
        rsize=60
    else:
        rsize=80
        
    if ratio_tw10.loc[ID, 'type0']==1:
        drawPieMarker(ax1,
                      xs=stations_t10.loc[ID, 'LON'],
                      ys=stations_t10.loc[ID, 'LAT'],
                      ratios=[1],
                      sizes=[rsize],
                      colors=[green])
        continue
        
    if ratio_excluded_tw10.loc[ID, 'type1'] == 1:
        ratios = [1]
        colors = [blue]
    elif ratio_excluded_tw10.loc[ID, 'type2'] == 0:
        ratios = [ratio_excluded_tw10.loc[ID, 'type1'],
                  ratio_excluded_tw10.loc[ID, 'others']]
        colors = [blue, gray]
    elif ratio_excluded_tw10.loc[ID, 'others'] == 0:
        ratios = [ratio_excluded_tw10.loc[ID, 'type1'],
                  ratio_excluded_tw10.loc[ID, 'type2']]
        colors = [blue, red]
    else:
        ratios = [
                  ratio_excluded_tw10.loc[ID, 'type1'], 
                  ratio_excluded_tw10.loc[ID, 'type2'],
                  ratio_excluded_tw10.loc[ID, 'others'],
                 ]
        colors = [blue, red, gray]
    drawPieMarker(ax1,
                  xs=stations_t10.loc[ID, 'LON'],
                  ys=stations_t10.loc[ID, 'LAT'],
                  ratios=ratios,
                  sizes=[rsize],
                  colors=colors)
ax1.set_title('Sounding Types in Precipitation Events (Tw)', fontsize=10)
ax1.set_title('(a)', loc='left',x=-0.06, fontsize=10)
sct0 = ax1.scatter(-130, 0, facecolor=green)
sct1 = ax1.scatter(-130, 0, facecolor=blue)
sct2 = ax1.scatter(-130, 0, facecolor=red)
sct3 = ax1.scatter(-130, 0, facecolor=gray)
ax1.legend( [sct0, sct1, sct2, sct3], 
           ['Type0', 'Type1','Type2','Others'],
          loc='lower left',
          fontsize=8)


ax2  = fig.add_axes([0.15, 0.165, 0.65, 0.3], projection=proj)
plot_basemap(ax2, img_extent, dlon, dlat)
stations_t10 = stations.loc[ratio_tw10_snow.index]
for ID in stations_t10.index:
    num = num_sounding_tw10_snow.loc[ID, 'total']
    if num<=50:
        rsize = 20
    elif num<=200:
        rsize=40
    elif num<=500:
        rsize=60
    else:
        rsize=80
        
    if ratio_tw10_snow.loc[ID, 'type0']==1:
        drawPieMarker(ax2,
                      xs=stations_t10.loc[ID, 'LON'],
                      ys=stations_t10.loc[ID, 'LAT'],
                      ratios=[1],
                      sizes=[rsize],
                      colors=[green])
        continue
        
    if ratio_excluded_tw10_snow.loc[ID, 'type1'] == 1:
        ratios = [1]
        colors = [blue]
    elif ratio_excluded_tw10_snow.loc[ID, 'type2'] == 0:
        ratios = [ratio_excluded_tw10_snow.loc[ID, 'type1'],
                  ratio_excluded_tw10_snow.loc[ID, 'others']]
        colors = [blue, gray]
    elif ratio_excluded_tw10_snow.loc[ID, 'others'] == 0:
        ratios = [ratio_excluded_tw10_snow.loc[ID, 'type1'],
                  ratio_excluded_tw10_snow.loc[ID, 'type2']]
        colors = [blue, red]
    else:
        ratios = [
                  ratio_excluded_tw10_snow.loc[ID, 'type1'], 
                  ratio_excluded_tw10_snow.loc[ID, 'type2'],
                  ratio_excluded_tw10_snow.loc[ID, 'others'],
                 ]
        colors = [blue, red, gray]
    drawPieMarker(ax2,
                  xs=stations_t10.loc[ID, 'LON'],
                  ys=stations_t10.loc[ID, 'LAT'],
                  ratios=ratios,
                  sizes=[rsize],
                  colors=colors)

ax2.set_title('Sounding Types in Snow Events (Tw)', fontsize=10)
ax2.set_title('(b)', loc='left', x=-0.06, fontsize=10)

sct11 = drawPieMarker(ax2, 30, 40, [1], [160], [blue])
sct12 = drawPieMarker(ax2, 30, 40, [1], [120], [blue])
sct13 = drawPieMarker(ax2, 30, 40, [1], [80], [blue])
sct14 = drawPieMarker(ax2, 30, 40, [1], [40], [blue])
legend2 = plt.legend([sct11, sct12, sct13, sct14],
                     ['>500', '200-500', '50-200', '<50'],
                     bbox_to_anchor=[1.38, .60],
                     frameon=False,
                     labelspacing=1)
ax2.add_artist(legend2)
ax2.text(1.08, 0.63, '$N_{soundings}$',transform=ax2.transAxes)

sct0 = ax2.scatter(-130, 0, facecolor=green)
sct1 = ax2.scatter(-130, 0, facecolor=blue)
sct2 = ax2.scatter(-130, 0, facecolor=red)
sct3 = ax2.scatter(-130, 0, facecolor=gray)
ax2.legend( [sct0, sct1, sct2, sct3], 
           ['Type0', 'Type1','Type2','Others'],
          loc='lower left',
          fontsize=8)

plt.savefig(figpath+'Figure3', dpi=300, bbox_inches='tight')
plt.savefig(figpath+'Figure3.eps',format='eps', dpi=300, bbox_inches='tight')


