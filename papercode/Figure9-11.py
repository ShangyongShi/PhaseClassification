#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:59:10 2024

Figure 9-11

@author: ssynj
"""


import pandas as pd
import numpy as np
# from function.evaluation import *
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker
import matplotlib.colors as colors
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
    
def classify(test, threshold):
    test0 = test[ test['type_ti']==0]
    test1 = test[ test['type_ti']==1]
    test2 = test[ test['type_ti']==2]
    
    pre_rain0, pre_snow0 = classify_type0(test0)
    pre_rain1, pre_snow1 = classify_type1(test1, threshold)
    pre_rain2, pre_snow2 = classify_type2(test2, threshold)
    
    pre_rain = pd.concat([pre_rain0, pre_rain1, pre_rain2])
    pre_snow = pd.concat([pre_snow0, pre_snow1, pre_snow2])   
    return pre_rain, pre_snow

def classify_type0(test):
    snow = test[test['tw']<=1.6]
    rain = test[test['tw']>1.6]
    return rain.wwflag, snow.wwflag

def type1_exp(x, m, t):
    return m * np.exp(t*x ) 

def classify_type1(test, threshold):
    '''
    Seperation for soundings with only one melting at the bottom
    '''
    coefs = {0.3: np.array([1.8352, -0.1688]),
            0.4: np.array([1.5738, -0.2053]),
            0.5: np.array([1.2957, -0.2748]),
            0.6: np.array([1.0060, -0.3635]),
            0.7: np.array([0.7358, -0.3635]),
            0.8: np.array([0.4, -0.4])
            }
    coefs = {
             0.3: np.array([1.68332365, -0.1811878 ]),
             0.4: np.array([1.42235443, -0.22139454]),
             0.5: np.array([1.19237535, -0.29651954]),
             0.6: np.array([0.93126144, -0.42526325]),
             0.7: np.array([0.92178506, -1.31780889]),
             0.8: np.array([0.4477541 , -2.14061253])}
    popt = coefs[threshold]
    pre_snow = test[test['ti'] <= type1_exp(test['PA_ti'], *popt)]
    pre_rain = test[test['ti'] >  type1_exp(test['PA_ti'], *popt)]
    return pre_rain, pre_snow


def type2_tanh(x,  b, c, d):
    return -18*(np.tanh(b*x -c))+d

def classify_type2(test, threshold):
    '''
    Separation for soundings with a melting layer and a refreezing layer
    '''
    coefs = {0.3: np.array([0.3216, -0.1059, 0.4968]),
            0.4: np.array([0.2770, -0.2366, -0.8472]),
            0.5: np.array([0.2819, -0.0287, -3.4724]),
            0.6: np.array([0.3112, -0.1421, -4.6392]),
            0.7: np.array([0.4217, -0.4273, -6.0914]),
            0.8: np.array([0.5647, -1.3234, -7.3206]) 
            }
    coefs = {0.3: np.array([ 0.11992169, -0.48402358,  7.40607351]),
             0.4: np.array([ 0.08766819,  0.0641349 , -3.59167715]),
             0.5: np.array([  0.14446052,   0.63535494, -14.0823996 ]),
             0.6: np.array([  0.21317597,   0.69167411, -16.86781474]),
             0.7: np.array([  0.29617753,   0.47365385, -18.36315865]),
             0.8: np.array([  0.39996308,  -0.11066325, -19.82783235])}

    ME = test['PA_ti'].values
    RE = abs(test['NA_ti'].values)
    
    pre_rain = test[test['ti']>  type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    pre_snow = test[test['ti']<= type2_tanh(np.log(3*ME/(2*RE)), *coefs[threshold])]
    return pre_rain, pre_snow
    
def print_metrics(pre_rain, pre_snow, printflag):
    '''
    print_metrics(pre_rain, pre_snow, printflag)
    '''
    TP = sum(pre_snow.wwflag==2)
    FP = sum(pre_snow.wwflag==1)
    P = len(pre_snow)
    TN = sum(pre_rain.wwflag==1)
    FN = sum(pre_rain.wwflag==2)
    N = len(pre_rain)
    
    metrics = {}
    
    metrics['accuracy'] = (TP+TN)/(P+N)
    if TP+FN == 0:
        metrics['recall'] = np.nan
    else:
        metrics['recall'] = TP/(TP+FN)
    
    if TP+FP ==0:
        metrics['precision'] = np.nan
        metrics['POFA'] = np.nan
    else:
        metrics['precision'] = np.divide(TP, (TP+FP))
        metrics['POFA'] = 1-metrics['precision']
        
    if FP+TN==0:
        metrics['POFD'] = np.nan
    else:
        metrics['POFD'] = FP/(FP+TN)
        
    if TP==0:
        metrics['f1score'] = np.nan
    else:
        metrics['f1score'] = 1 /((TP+FN)/TP+(TP+FP)/TP) 
     
    metrics['POD'] = metrics['recall']
    
    if TP+FP+FN==0:
        metrics['CSI'] = np.nan
    else:
        metrics['CSI'] = TP/(TP+FP+FN)
      
    if ( (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN) ) ==0:
        metrics['HSS'] = np.nan
    else:
        metrics['HSS'] = (2*(TP*TN-FP*FN)) / ( (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN) )   
      
    metrics['TSS'] = metrics['POD'] - metrics['POFD']
    
    
    if printflag:
        strform = 'True positive: %d | False positive: %d | P_PRE:%d\n' +\
                  'False negative: %d | True negative: %d | N_PRE:%d \n' +\
                   'P_OBS: %d | N_OBS: %d\n | TOTAL: %d \n\n' +\
                  'Accuracy: %5.3f \n' +\
                  'Recall: %5.3f \n' +\
                  'Precision: %5.3f \n' +\
                  'F1Score: %5.3f \n' +\
                  'POD (Probability of Detection): %5.3f \n' +\
                  'POFA (False Alarm Ratio): %5.3f \n' +\
                  'POFD (Probability of false detection, False Alarm Rate): %5.3f \n' +\
                  'CSI (Critical Success Index): %5.3f \n' +\
                  'HSS (Heidke Skill Score): %5.3f \n' +\
                     'TSS (True Skill Statistics): %5.3f \n'
        print(strform %(TP, FP, TP+FP, FN, TN, TN+FN, TP+FN, FP+TN, P+N, 
                        metrics['accuracy'], metrics['recall'], 
                        metrics['precision'], metrics['f1score'],
                        metrics['POD'], metrics['POFA'], metrics['POFD'],
                        metrics['CSI'], metrics['HSS'], metrics['TSS']))
    return metrics

#%%
datapath = '../data/'
figpath = '../figure/'
threshold = 0.5

# df = pd.read_csv(datapath + 'final_all_cleaned_data.txt', index_col=0, dtype={'ID':str})
# de1 = pd.read_csv(datapath + 'type1_ti_de.txt', index_col=0, dtype={'ID':str})
# de2 = pd.read_csv(datapath + 'type2_ti_de.txt', index_col=0, dtype={'ID':str})

# df.drop(index=de1.index, inplace=True)
# df.drop(index=de2.index, inplace=True)

# test = df[  ~df.t.isna() & 
#             ~df.td.isna() & 
#             ~df.p.isna() & 
#             ~df.lapse_rate_tw.isna() &
#             ~df.lapse_rate_t.isna() &
#             (df.type_ti>0)]

test = pd.read_csv(datapath+'Figure9/Figure9_testdata.txt')

IDs = np.sort(test.ID.unique())
nums = pd.DataFrame(index=IDs, 
                      columns=['ndata', 'nrain', 'nsnow'])
scores = pd.DataFrame(index=IDs, 
                      columns=['accuracy', 'f1score', 'FARatio',
                               'POD', 'FAR', 'CSI', 'HSS', 'TSS'])
bias_energy = pd.DataFrame(index=IDs, columns=['absolute'])
bias_energy.sort_index(inplace=True)
for ID in IDs:
    test_station = test[test.ID == ID]
    
    pre_rain, pre_snow = classify(test_station, threshold)
    
    
    nums.loc[ID, 'ndata'] = len(test_station)
    nums.loc[ID, 'nrain'] = sum(test_station.wwflag==1)
    nums.loc[ID, 'nsnow'] = sum(test_station.wwflag==2)
    metrics = print_metrics(pre_rain, pre_snow, False)
    scores.loc[ID, 'accuracy':'TSS'] = [
        metrics['accuracy'], metrics['f1score'], metrics['POFA'],
        metrics['POD'], metrics['POFD'], 
        metrics['CSI'], metrics['HSS'], metrics['TSS']
        ]
    
    n_pre_rain = len(pre_rain)
    n_pre_snow = len(pre_snow)
    snowp_pre = n_pre_snow/len(test_station)
    snowp_obs = len(test_station[test_station.wwflag==2])/len(test_station)
    bias_energy.loc[ID, 'absolute'] = snowp_pre-snowp_obs 
    
    
    
stations = pd.read_csv('../data/NCEP_IGRA_collocated_stations_cleaned.txt')
stations.set_index('NCEP_ID', inplace=True)
# lat and lon of the stations
bias_loc = stations.loc[bias_energy.index]
bias_loc = bias_loc[~bias_loc.index.duplicated()]
    
bias_energy['lon'] = bias_loc['LON']
bias_energy['lat'] = bias_loc['LAT']
# bias_energy.to_csv('../data/bias_energy_'+str(threshold*100)+'.txt')
  
# scores.to_csv('../data/scores_energy_'+str(threshold*100)+'.txt')

#%%
# bias_energy = pd.read_csv('../data/bias_energy.txt', dtype={'ID':str}, index_col=0)
# nums = pd.read_csv(datapath+'num_type1_type2.txt', dtype={'ID':str}, index_col=0)

#%% ------------------------
bias = pd.read_csv('../data/Figure2/snow_fraction_bias_temp_and_probsnow.txt', dtype={'ID':str}, index_col=0)
bias.drop(columns=[ 'lon', 'lat', 'elev'], inplace=True)

# %%
# set projection
leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 85)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15
proj = ccrs.PlateCarree(central_longitude=0)

# set colormap
rgb = pd.read_csv( '../data/colormap/diff_16colors.txt', 
                  delim_whitespace=True, header=None).values/255
rgb = rgb[1:-1] #14 colors
cmap = colors.LinearSegmentedColormap.from_list('mymap', rgb[1:-1], len(rgb)-2)
cmap.set_under(rgb[0])
cmap.set_over(rgb[-1])

# set boundaries
#%%
'''
Figure 9 bias and difference in absolute bias
'''
# ax1  = fig.add_axes([0.1, 0.18, 0.75, 0.75], projection=proj)
# position1 = fig.add_axes([0.095, 0.08, 0.765, 0.03])
    
fig = plt.figure(figsize=(11/2.54, 14/2.54), dpi=1000)  
ax1  = fig.add_axes([0.15, 0.65, 0.7, 0.3], projection=proj)
ax2  = fig.add_axes([0.15, 0.12, 0.7, 0.3], projection=proj)
position1 = fig.add_axes([0.11, 0.58, 0.75, 0.012])
position2 = fig.add_axes([0.11, 0.05, 0.75, 0.012])
plot_basemap(ax1, img_extent, dlon, dlat)
plot_basemap(ax2, img_extent, dlon, dlat)


bounds = [-6, -5,-4, -3, -2, -1, -0.1, 0, 0.1, 2, 4, 8, 12, 20] # in %
bounds=[-16, -8, -4, -2, -1,-0.5, 0, 0.5,1, 2, 4, 8,  16] 
boundlabels=bounds
norms = colors.BoundaryNorm(boundaries=bounds, ncolors=len(rgb)-2)

sct1= ax1.scatter(bias_energy['lon'], bias_energy['lat'],
                  c=bias_energy['absolute']*100,
                  s=20,
                  cmap=cmap,
                  norm=norms)
ax1.set_title('Energy Method', fontsize=12)
cb = plt.colorbar(sct1, cax=position1 ,orientation='horizontal', fraction=.1, extend='both')
cb.ax.tick_params(labelsize=10)
cb.set_ticks(bounds)
cb.set_ticklabels(bounds)
cb.ax.set_xlabel('Bias in Conditional Probability of Snow (%)')



diff = bias.copy()
for col in diff.columns:
    diff[col] = abs(bias_energy['absolute'])-abs(diff[col])

diff = diff.loc[nums.ndata>=10] # more than 10 events?

bounds = [ -6,-5, -4, -3, -2, -1, 0,  1, 2, 3, 4, 5, 6] 
boundlabels=bounds
norms = colors.BoundaryNorm(boundaries=bounds, ncolors=len(rgb)-2)

plot_basemap(ax2, img_extent, dlon, dlat)
sct1= ax2.scatter(bias_energy.loc[diff.index, 'lon'], 
                  bias_energy.loc[diff.index, 'lat'],
                  c=diff['probsnow_tw']*100, norm=norms,
                  s=20,
                  cmap=cmap)
ax2.set_title('Energy - Probsnow', fontsize=12)

cb = plt.colorbar(sct1, cax=position2 ,orientation='horizontal', fraction=.1, extend='both')
cb.ax.tick_params(labelsize=10)
cb.set_ticks(bounds)
cb.set_ticklabels(bounds)
cb.ax.set_xlabel('Difference of Absolute Bias (%)')

ax1.set_title('(a)', loc='left', fontsize=12)
ax2.set_title('(b)', loc='left', fontsize=12)

ax1.text(-166, 28, r'$\overline{bias}$'+'=%.2f'% (bias_energy.absolute.mean()*100)+'%')

plt.savefig(figpath+'Figure9_'+str(int(threshold*100)), dpi=1000, bbox_inches='tight')
plt.savefig(figpath+'Figure9_'+str(int(threshold*100))+'.pdf', format='pdf', bbox_inches='tight')

# plt.savefig(figpath+'Figure_diff_absbias_'+str(int(threshold*100)), dpi=300, bbox_inches='tight')
# plt.savefig(figpath+'Figure_diff_absbsias_'+str(int(threshold*100))+'.eps', format='eps', dpi=300, bbox_iches='tight')

#%%
'''
Figure 10
'''
# scores= pd.read_csv(datapath+'/scores_energy_'+str(threshold*100)+'.txt',
#                     index_col=0)

stations = pd.read_csv(datapath+'NCEP_IGRA_collocated_stations_cleaned.txt')
stations.set_index('NCEP_ID', inplace=True)
# lat and lon of the stations
scores = scores.loc[nums.ndata>=10]

wbgyr = pd.read_csv('../data/whiterainbow.txt', 
                    delim_whitespace=True, header=None)/255
bgyr = wbgyr[20:]
mycmap = colors.ListedColormap(bgyr.values, name='mycmap')


proj = ccrs.PlateCarree(central_longitude=0)
leftlon, rightlon, lowerlat, upperlat = (-170, -50, 25, 85.5)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
dlon, dlat = 30, 15

fig = plt.figure(figsize=(19/2.54, 14/2.54), dpi=1000)  
ax1  = fig.add_axes([0.15, 0.58, 0.35, 0.3], projection=proj)
ax2  = fig.add_axes([0.60, 0.58, 0.35, 0.3], projection=proj)
ax3  = fig.add_axes([0.15, 0.13, 0.35, 0.3], projection=proj)
ax4  = fig.add_axes([0.60, 0.13, 0.35, 0.3], projection=proj)
position1 = fig.add_axes([0.15, 0.53, 0.35, 0.01])
position2 = fig.add_axes([0.60, 0.53, 0.35, 0.01])
position3 = fig.add_axes([0.15, 0.08, 0.35, 0.01])
position4 = fig.add_axes([0.60, 0.08, 0.35, 0.01])

plot_basemap(ax1, img_extent, dlon, dlat)
plot_basemap(ax2, img_extent, dlon, dlat)
plot_basemap(ax3, img_extent, dlon, dlat)
plot_basemap(ax4, img_extent, dlon, dlat)

sct1 = ax1.scatter(stations.loc[scores.index, 'LON'], stations.loc[scores.index, 'LAT'], c=scores.accuracy, s=20, cmap=mycmap, vmin = 0.8)
sct2 = ax2.scatter(stations.loc[scores.index, 'LON'], stations.loc[scores.index, 'LAT'], c=scores.HSS, s=20, cmap=mycmap, vmin=0.3, vmax=0.9)
sct3 = ax3.scatter(stations.loc[scores.index, 'LON'], stations.loc[scores.index, 'LAT'], c=scores.POD, s=20, cmap=mycmap, vmin=0.3, vmax=0.9)
sct4 = ax4.scatter(stations.loc[scores.index, 'LON'], stations.loc[scores.index, 'LAT'], c=scores.FAR, s=20, cmap=mycmap, vmin=0.01, vmax=0.09)


cb1 = plt.colorbar(sct1, cax=position1, orientation='horizontal', extend='min')
cb2 = plt.colorbar(sct2, cax=position2, orientation='horizontal', extend='both')
cb3 = plt.colorbar(sct3, cax=position3, orientation='horizontal', extend='both')
cb4 = plt.colorbar(sct4, cax=position4, orientation='horizontal', extend='both')

cb2.set_ticks(np.arange(0.3, 0.91, 0.2))
cb3.set_ticks(np.arange(0.3, 0.91, 0.2))
cb4.set_ticks(np.arange(0.01, 0.091, 0.02))

ax1.set_title('Accuracy', fontsize=12)
ax3.set_title('POD', fontsize=12)
ax4.set_title('FAR', fontsize=12)
ax2.set_title('HSS', fontsize=12)

ax1.set_title('(a)', loc='left', fontsize=12)
ax2.set_title('(b)', loc='left', fontsize=12)
ax3.set_title('(c)', loc='left', fontsize=12)
ax4.set_title('(d)', loc='left', fontsize=12)
plt.suptitle('Energy Method', y=0.94, x=0.52, fontsize=14)

plt.savefig(figpath+'Figure10_'+str(int(threshold*100)), bbox_inches='tight', dpi=1000)
plt.savefig(figpath+'Figure10_'+str(int(threshold*100))+'.pdf', format='pdf', bbox_inches='tight')

#%%
# --------------

scores_probsnow = pd.read_csv(datapath+'Figure11/'+str(int(threshold*100))+'_TwProbsnow_score_map.txt',
                              index_col=0)
scores_probsnow = scores_probsnow.loc[nums.ndata>=10]


score = scores - scores_probsnow

# if difference=0, don't plot the dot
diff = {}
strings = ['accuracy','POD', 'FAR', 'HSS', 'TSS']
for var in strings:
    score.loc[score[var]==0, var]=np.nan
    diff[var] = score.loc[~score[var].isna(), var]
    
dfcmap = pd.read_csv(r'/Users/ssynj/Dropbox/Research/MyColormap/NCV_bluered.txt', delim_whitespace=True, header=None)

# dfcmap = pd.read_csv('diff_16colors.txt', header=None, dtype=int, delim_whitespace=True).values/255
dfcmap = pd.read_csv(datapath+'/colormap/darkbluedarkred.txt', header=None).values/255
cmap = colors.ListedColormap(dfcmap[1:-1])
cmap.set_under(dfcmap[0, :])
cmap.set_over(dfcmap[-1, :])

fig = plt.figure(figsize=(19/2.54, 14/2.54), dpi=1000)  
ax1  = fig.add_axes([0.15, 0.58, 0.35, 0.3], projection=proj)
ax2  = fig.add_axes([0.60, 0.58, 0.35, 0.3], projection=proj)
ax3  = fig.add_axes([0.15, 0.13, 0.35, 0.3], projection=proj)
ax4  = fig.add_axes([0.60, 0.13, 0.35, 0.3], projection=proj)
position1 = fig.add_axes([0.15, 0.53, 0.35, 0.01])
position2 = fig.add_axes([0.60, 0.53, 0.35, 0.01])
position3 = fig.add_axes([0.15, 0.08, 0.35, 0.01])
position4 = fig.add_axes([0.60, 0.08, 0.35, 0.01])
plot_basemap(ax1, img_extent, dlon, dlat)
plot_basemap(ax2, img_extent, dlon, dlat)
plot_basemap(ax3, img_extent, dlon, dlat)
plot_basemap(ax4, img_extent, dlon, dlat)


bounds1 = [-0.01 , -0.008, -0.006, -0.004, -0.002, -0.001, 0., 0.001   ,  0.002,  0.004, 0.006,  0.008,  0.01 ]
boundlabels=bounds1
norms = colors.BoundaryNorm(boundaries=bounds1, ncolors=len(dfcmap)-2)

sct1 = ax1.scatter(stations.loc[diff['accuracy'].index, 'LON'],  stations.loc[diff['accuracy'].index, 'LAT'],  c=diff['accuracy'],  cmap=cmap, s=20, vmin=-0.1, vmax=0.1)
sct2 = ax2.scatter(stations.loc[diff['HSS'].index, 'LON'],   stations.loc[diff['HSS'].index, 'LAT'],   c=diff['HSS'],   cmap=cmap, s=20, vmin=-0.4, vmax=0.4)
sct3 = ax3.scatter(stations.loc[diff['POD'].index, 'LON'],  stations.loc[diff['POD'].index, 'LAT'],  c=diff['POD'],  cmap=cmap, s=20, vmin=-0.4, vmax=0.4)
sct4 = ax4.scatter(stations.loc[diff['FAR'].index, 'LON'],    stations.loc[diff['FAR'].index, 'LAT'],    c=diff['FAR'],    cmap=cmap, s=20, vmin=-0.1, vmax=0.1)

cb1 = plt.colorbar(sct1, cax=position1, orientation='horizontal', extend='both')
cb2 = plt.colorbar(sct2, cax=position2, orientation='horizontal', extend='both')
cb3 = plt.colorbar(sct3, cax=position3, orientation='horizontal', extend='both')
cb4 = plt.colorbar(sct4, cax=position4, orientation='horizontal', extend='both')

cb2.set_ticks(np.arange(-0.4, 0.41, 0.2))
#cb4.set_ticks(np.arange(-0.3, 0.31, 0.15))

ax1.set_title('Accuracy Difference', fontsize=12)
ax3.set_title('POD Difference', fontsize=12)
ax4.set_title('FAR Difference', fontsize=12)
ax2.set_title('HSS Difference', fontsize=12)

ax1.set_title('(a)', loc='left', fontsize=12)
ax2.set_title('(b)', loc='left', fontsize=12)
ax3.set_title('(c)', loc='left', fontsize=12)
ax4.set_title('(d)', loc='left', fontsize=12)
plt.suptitle('Energy Method minus Probsnow Method', x=0.52, y=0.94, fontsize=14)

plt.savefig(figpath+'Figure11_'+str(int(threshold*100)), bbox_inches='tight', dpi=1200)
plt.savefig(figpath+'Figure11_'+str(int(threshold*100))+'.pdf', format='pdf', bbox_inches='tight')    
    





