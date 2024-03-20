#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:13:49 2024

Figure 6

@author: ssynj
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from skimage import measure
from scipy.optimize import curve_fit

from watervapor import td2rh

def conp_2d(rain, snow, xvar, yvar, params):
    '''
    on the plot, column is x axis, row is y axis
    
    Input:
        rain, snow: DataFrames
        xvar, yvar: str
        params:  xmin, xmax, xbinsize, ymin, ymax, ybinsize
    Output:
        conp, num_rain, num_snow
    '''
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
    
    # column is x axis, row is y axis on the plot
    num_rain = pd.DataFrame(data=0, 
                            index=np.arange(ymin, ymax, ybinsize)+ybinsize/2, 
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    num_snow = pd.DataFrame(data=0, 
                            index=np.arange(ymin, ymax, ybinsize)+ybinsize/2, 
                            columns=np.arange(xmin, xmax, xbinsize)+xbinsize/2)
    for row in num_rain.index:
        for col in num_rain.columns:
            num_rain.loc[row, col] += ((rain[yvar]>row-ybinsize/2) & 
                                       (rain[yvar]<=row+ybinsize/2) & 
                                       (rain[xvar]>col-xbinsize/2) & 
                                       (rain[xvar]<=col+xbinsize/2)).sum()
            num_snow.loc[row, col] += ((snow[yvar]>row-ybinsize/2) & 
                                       (snow[yvar]<=row+ybinsize/2) &
                                       (snow[xvar]>col-xbinsize/2) & 
                                       (snow[xvar]<=col+xbinsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)
    return conp, num_rain, num_snow

def LDA_boundary_line(x_r, x_s):
    '''
    x_r, x_s: two columns indicate two groups
    Output: slope and intercept of the boundary line
    '''
    mu_r = x_r.mean(axis=0)
    mu_s = x_s.mean(axis=0)

    cov_r = np.cov(x_r, rowvar=False)
    cov_s = np.cov(x_s, rowvar=False)
    W = ((len(x_r)-1)*cov_r + (len(x_s)-1)*cov_s) / (len(x_r)+len(x_s)-2) # pooled within group covariance
    W_1 = np.linalg.inv(W)


    # the boundary line is perpendicular to this line,
    # crossing the midpoint of the centroids
    line = np.dot(W_1, (mu_s - mu_r))     
    midpoint = (mu_r+mu_s)/2
    
    slope = -line[0]/line[1]
    intercept = midpoint[1]-slope*midpoint[0]
    return slope, intercept

def fit_LDA_for_range(df, box):
    '''
    Input: df, box (xmin, xmax, ymin, ymax)
    Output: slope, intercept for LDA boundary line
    '''
    xmin, xmax, ymin, ymax = box
    
    # LDA separation line for x>xmin1
    rain = df[df.wwflag==1]
    snow = df[df.wwflag==2]
    rain1 = rain[(rain[PA]>=xmin) & (rain[PA]<=xmax) &
                 (rain[NA]>=ymin) & (rain[NA]<=ymax)]
    snow1 = snow[(snow[PA]>=xmin) & (snow[PA]<=xmax) &
                 (snow[NA]>=ymin) & (snow[NA]<=ymax)]
    
    x_r1 = np.array((pd.concat([rain1[PA], rain1[NA]], axis=1)))
    x_s1 = np.array((pd.concat([snow1[PA], snow1[NA]], axis=1)))

    slope1, intercept1 = LDA_boundary_line(x_r1, x_s1)
    return slope1, intercept1


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

blue = np.array([14, 126, 191])/255
red = np.array([235, 57, 25])/255
 

datapath = '../data/'
figpath = '../figure/'
PA = 'PA_ti'
NA = 'NA_ti'
#%% --------------------------------------------------------------------
# read data
type2_ti_de = pd.read_csv(datapath+'type2_ti_de.txt', index_col=0)
type2_ti_ev = pd.read_csv(datapath+'type2_ti_ev.txt', index_col=0)
type2_ti_de[NA] *= -1
type2_ti_ev[NA] *= -1
rain = type2_ti_de[type2_ti_de.wwflag==1]
snow = type2_ti_de[type2_ti_de.wwflag==2]


# fit LDA
slope, intercept = fit_LDA_for_range(type2_ti_de, [0, 500, 0, 500])
print('LDA boundary: %.3f *TwME + %.3f - TwRE = 0' % (slope, intercept))


''' Scatter plot and LDA '''
#fig, ax = plot_type2_scatter_LDA(rain, snow, PA, NA, slope, intercept)
fig = plt.figure(figsize=(5, 4), dpi=1200)
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
s1 = ax.scatter(rain[PA], rain[NA], color=blue, marker='o', linewidth=0.8, facecolor='none', s=10, alpha=0.8)
s2 = ax.scatter(snow[PA], snow[NA], color=red, marker='^',linewidth=0.8,facecolor='none', s=14, alpha=0.6)

xs = np.arange(0, 500+0.1, 0.1)
l1, = ax.plot(xs, (slope*xs)+intercept,'--', color=[.26,.26,.26], linewidth=3) 

lg1 = ax.legend([s1, s2],['rain', 'snow'], loc='upper right' )
ax.text(190, 220, 'TiRE - %.2f TiME + %.2f = 0' % (slope, -intercept))

ax.set_xlim([0, 500])
ax.set_ylim([0, 500])
ax.set_xlabel('TwME (J/kg)')
ax.set_ylabel('TwRE (J/kg)')
ax.set_title('Type 2 Sounding\nLDA Separation Between Rain and Snow')

plt.savefig(figpath+'Figure6',  bbox_inches='tight', dpi=1200)
plt.savefig(figpath+'Figure6.pdf', format='pdf')


#%%


# merge ME and RE
npa = 3
nna = 2
rain['net'] = np.log(npa*rain['PA_ti'])-np.log(nna*rain['NA_ti'])
snow['net'] = np.log(npa*snow['PA_ti'])-np.log(nna*snow['NA_ti'])

params = [-8, 8, 2,
           -10, 10, 2]
conp, num_rain, num_snow = conp_2d(rain, snow, 'ti', 'net', params)
conp1 = conp.mask(num_rain+num_snow<10)


# ----------------
def running_conp(rain, snow, xvar, yvar, params):
    '''
    calculate the running counts and probability
    the summing grids will overlap
    
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
    
    '''
    # xvar, yvar = 'tw', 'PA'
    xmin, xmax, xbinsize, ymin, ymax, ybinsize = params
        # column: xvar
        # row: yvar
        # when plotting, we plot the transpose of the matrix
    # column is x axis, row is y axis on the plot
    num_rain = pd.DataFrame(data=0, 
                            index=np.round(np.arange(ymin, ymax+0.1, 0.1),decimals=1), 
                            columns=np.round(np.arange(xmin, xmax+0.1, 0.1), decimals=1))
    num_snow = pd.DataFrame(data=0, 
                            index=np.round(np.arange(ymin, ymax+0.1, 0.1),decimals=1), 
                            columns=np.round(np.arange(xmin, xmax+0.1, 0.1), decimals=1))
    for row in num_rain.index:
        for col in num_rain.columns:
            num_rain.loc[row, col] += ((rain[yvar]>row-ybinsize/2) & 
                                       (rain[yvar]<=row+ybinsize/2) & 
                                       (rain[xvar]>col-xbinsize/2) & 
                                       (rain[xvar]<=col+xbinsize/2)).sum()
            num_snow.loc[row, col] += ((snow[yvar]>row-ybinsize/2) & 
                                       (snow[yvar]<=row+ybinsize/2) &
                                       (snow[xvar]>col-xbinsize/2) & 
                                       (snow[xvar]<=col+xbinsize/2)).sum()
    conp = num_snow/(num_rain+num_snow)
    return conp, num_rain, num_snow


def running_mean(conp2, d):
    '''
    calculate the running mean within d*d box for conp2
    '''
    cut = np.round(d/2, decimals=1)
    newcol = np.round(np.arange(conp2.columns[0]+cut, conp2.columns[-1]-cut+0.1, 0.1), decimals=1)
    newrow = np.round(np.arange(conp2.index[0]+cut, conp2.index[-1]-cut+0.1, 0.1), decimals=1)
    conp_smooth = pd.DataFrame(data=np.nan, index=newrow, columns=newcol)

    for row in newrow:
        for col in newcol:
            tmp = conp2.loc[row-cut:row+cut, col-cut:col+cut]
            conp_smooth.loc[row, col] = tmp.sum().sum()/ (tmp.shape[0]*tmp.shape[1])
    conp_smooth[conp_smooth==0] = np.nan
    return conp_smooth 
    

params = [-8, 2, 4,
            -10, 10, 4]
conp2, num_rain2, num_snow2 = running_conp(rain, snow, 'ti', 'net', params)
conp2 = conp2.mask(num_rain2+num_snow2<10)
params = {'xlim': [-9, 5],
          'ylim': [-8, 1],
          'xlabel': 'ln('+str(npa)+'TiME/'+str(nna)+'TiRE)',
          'ylabel':'Ti ('+chr(176)+'C)'}
#plot_type2_contour(conp2, params)

# running mean
conp_smooth = running_mean(conp2, 2)
num_rain_smooth = running_mean(num_rain, 2)
num_snow_smooth = running_mean(num_snow, 2)
conp_smooth1 = conp_smooth.mask(num_rain_smooth+num_snow_smooth<10)
#plot_type2_contour(conp_smooth1, params)

LUT = conp_smooth1.loc[-7:3, -7:1]
params = {'xlim': [-7, 3],
          'ylim': [-7, 1],
          'xlabel': 'ln('+str(npa)+'TiME/'+str(nna)+'TiRE)',
          'ylabel':'Ti ('+chr(176)+'C)'}
plot_type2_contour(LUT, params)

''' 
fit with '''
def get_contour_values(df):
    '''
    cn is the returned contour 
    '''
    cn = plt.contour(df.index, df.columns, df.T, levels=np.arange(0, 1.1, 0.1))
    plt.close()
    contours = []
    idx = 0
    # match contour value with corresponding coordinates
    for cc, vl in zip(cn.collections, cn.levels):
        for pp in cc.get_paths():
            paths = {}
            paths['id'] = idx
            paths["value"] = float(vl)
            
            v = pp.vertices
            x = v[:, 0]
            y = v[:, 1]
            
            paths["x"] = x
            paths["y"] = y
            contours.append(paths)
            idx += 1
            
    # merge arrays with the same contour value
    xs, ys = {}, {}
    for value in cn.levels:
        x, y = [], []
        for contour in contours:
            if contour['value'] == float(value):
                x+=list(contour['x'])
                y+=list(contour['y'])
        
        xy = np.array([x, y])
        xy = xy[:, xy[0,:].argsort()]
        
        key = np.round(value, decimals=1)
        if np.size(xy)>0:  
            xy = np.unique(xy, axis=1)
        
            xs[key] = xy[0, :]
            ys[key] = xy[1, :]
        else:
            xs[key], ys[key] = [], []
    return xs, ys

from sklearn.metrics import r2_score

[xs, ys] = get_contour_values(LUT)
# =============================================================================
# 
# fig = plt.figure(figsize=(5, 4), dpi=300)
# ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
# for value in np.arange(0.3, 0.81, 0.1):
#     key = np.round(value, decimals=1)
#     ax.plot(xs[key], ys[key],'.', color='tab:red')
# =============================================================================
    
xs[0.3], ys[0.3] = xs[0.3][:-20], ys[0.3][:-20] 
xs[0.4], ys[0.4] = xs[0.4][:-23], ys[0.4][:-23] 
xs[0.5], ys[0.5] = xs[0.5][:-12], ys[0.5][:-12] 

def type2_tanh4(x, a, b, c, d):
    return a*(np.tanh(b*x -c))+d
initial_guess = [ -18, 0, 0, 0]
key = np.round(0.5, decimals=1)
X, Y = xs[key], ys[key]
coef, _ = curve_fit(type2_tanh4, X, Y, p0=initial_guess)
print(coef)
    
def type2_tanh(x,  b, c, d):
    return -18*(np.tanh(b*x -c))+d

model = type2_tanh
coefs = {}

initial_guess = [ 0, 0, 0]
for value in np.arange(0.3, 0.81, 0.1):
    key = np.round(value, decimals=1)
    X, Y = xs[key], ys[key]
    coef, _ = curve_fit(model, X, Y, p0=initial_guess)
    
    print(key, r2_score(Y, model(X, *coef)))
    coefs[key] = coef
# =============================================================================
# 
# fig = plt.figure(figsize=(5, 4), dpi=300)
# ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
# xxs = np.arange(-10, 7.1, 0.1)
# for value in np.arange(0.3, 0.81, 0.1):
#     key = np.round(value, decimals=1)
#     coef = coefs[key]
#     ax.plot(xs[key], ys[key],'.', color='tab:red')
#     ax.plot(xxs, type2_tanh(xxs, *coef), '.', color='tab:blue')
# ax.set_ylim([-10, 1])
#     
#     
# fig = plt.figure(figsize=(5, 4), dpi=300)
# ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
# params = {'xlim': [-9, 5],
#           'ylim': [-8, 1],
#           'xlabel': 'ln('+str(npa)+'TwME/'+str(nna)+'TwRE)',
#           'ylabel':'Tw ('+chr(176)+'C)'}
# fig, ax = plot_type2_contour(conp1, params)
# xxs = np.arange(-10, 7.1, 0.1)
# for value in np.arange(0.3, 0.81, 0.1):
#     key = np.round(value, decimals=1)
#     coef = coefs[key]
#     ax.plot(xxs, model(xxs, *coef), color='tab:red')
# ax.set_ylim([-10, 5])
# fig.savefig(figpath+'type2_expfit.png', bbox_inches='tight', dpi=300)
# =============================================================================


'''Evaluate'''

from evaluate import evaluate_model, evaluate_probsnow
test = type2_ti_ev[~type2_ti_ev.t.isna() & 
                ~type2_ti_ev.td.isna() & 
                ~type2_ti_ev.p.isna() & 
                ~type2_ti_ev.lapse_rate_t.isna() &
                ~type2_ti_ev.lapse_rate_ti.isna()]

test['net'] = np.log(npa*test['PA_ti'])-np.log(nna*test['NA_ti'])


threshold = np.round(0.5, decimals=1)
print('----Type 2, net energy, ti'+str(threshold))
evaluate_model(test, 'net', 'ti', type2_tanh, coefs[threshold])


threshold = np.round(0.7, decimals=1)
print('----Type 2, net energy, ti '+str(threshold))
evaluate_model(test, 'net', 'ti', type2_tanh, coefs[threshold])

threshold = np.round(0.8, decimals=1)
print('----Type 2, net energy, '+str(threshold))
evaluate_model(test, 'net', 'tw', type2_tanh, coefs[threshold])

#%%

''' Paper figures'''

cm = 1/2.54  # centimeters in inches
# fig = plt.figure(figsize=(19*cm, 9*cm), dpi=600)
# ax = fig.add_axes([0.05, 0.1, 0.42, 0.75])
# ax2 = fig.add_axes([0.56, 0.1, 0.38, 0.75])

fig = plt.figure(figsize=(21*cm, 9*cm), dpi=1200)

ax = fig.add_axes([0.07, 0.15, 0.35, 0.68])
ax2 = fig.add_axes([0.53, 0.15, 0.35, 0.68])
cbax = fig.add_axes([0.91, 0.15, 0.01, 0.68])

# subplot 1 scatter
params = {'xlim': [-10, 10],
          'ylim': [-7, 1],
          'xlabel': 'ln('+str(npa)+'TiME/'+str(nna)+'TiRE)',
          'ylabel': 'Ti ('+chr(176)+'C)',
          'title': '(a)'}
xvar, yvar = 'net', 'ti'
s1 = ax.scatter(rain[xvar], rain[yvar], color=blue, marker='o', facecolor='none', linewidth=0.5, s=8, alpha=0.8)
s2 = ax.scatter(snow[xvar], snow[yvar], color=red, marker='^',facecolor='none', linewidth=0.5,s=12, alpha=0.6)
lg1 = ax.legend([s1, s2],
                ['rain', 'snow'], loc='lower right')
ax.set_ylim(params['ylim'])
ax.set_ylabel(params['ylabel'])
ax.set_xlim(params['xlim'])
ax.set_xlabel(params['xlabel'])
ax.set_title(params['title'], loc='left')
    
ax.plot([0, 0], [-8, 3], 'k', color='0.4')
ax.plot([-10, 10], [0, 0], 'k', color='0.4')
xxs = np.arange(-10, 10.1, 0.1)
style = {0.3:(0, (1, 1)), 0.5:'solid', 0.7:(0, (5, 1))}
for value in np.arange(0.3, 0.81, 0.2):
    key = np.round(value, decimals=1)
    coef = coefs[key]
    ax.plot(xxs, model(xxs, *coef), color='k', linestyle=style[key])
ax.text(-4, -6, '70%', fontweight='bold')
ax.text(0.5, -4.7, '50%', fontweight='bold')
ax.text(2.5, -3.7, '30%', fontweight='bold')

#fig.savefig(figpath + 'type2_separation_ti.png', bbox_inches='tight', dpi=300)
#fig.savefig(figpath + 'type2_separation.eps',format='eps', bbox_inches='tight')

# subplot2
params = {'xlim': [-7, 3],
          'ylim': [-7, 1],
          'xlabel': 'ln('+str(npa)+'TiME/'+str(nna)+'TiRE)',
          'ylabel':'Ti ('+chr(176)+'C)',
          'title':'(b)'}
# shadings
pc = ax2.contourf(conp2.index, conp2.columns, conp2.T, 
                 cmap='jet', 
                 origin='lower', levels=np.arange(0, 1.1, 0.1))
# contours
ct = ax2.contour(conp2.index, conp2.columns, conp2.T, 
                colors='k', linewidths=0.5,
                origin='lower', levels=np.arange(0, 1.1, 0.1))
ax2.clabel(ct, ct.levels, inline=True, fmt='%3.1f', fontsize=10)
cb = plt.colorbar(pc, cax=cbax, 
                  label='Conditional Probability of Snow')
ax2.set_ylim(params['ylim'])
ax2.set_ylabel(params['ylabel'])
ax2.set_xlim(params['xlim'])
ax2.set_xlabel(params['xlabel'])
    
for value in np.arange(0.3, 0.81, 0.1):
    key = np.round(value, decimals=1)
    coef = coefs[key]
    ax2.plot(xxs, model(xxs, *coef),'--',color='tab:red')
ax2.set_title(params['title'], loc='left')
fig.suptitle('Type 2 Sounding', x=0.52)

fig.savefig(figpath+'Figure7.png', bbox_inches='tight', dpi=1200)
fig.savefig(figpath+'Figure7.pdf', format='pdf')



#%%
'''figure sounding example'''
# find examples

f1 = 'SKA_USM00072786_19881206120000'
f2 = 'FSI_USM00072355_19790206180000'
f3 = '72694_USM00072694_20081222000000'
f4 = '71867_CAM00071867_19851221120000'

df1 = pd.read_csv(datapath+'sounding/'+f1+'.txt') # rain, positive energy ratio 
df2 = pd.read_csv(datapath+'sounding/'+f2+'.txt') # snow, large negative, 70273_USM00070273_20181111120000
df3 = pd.read_csv(datapath+'sounding/'+f3+'.txt') # rain, negetive energy ratio
df4 = pd.read_csv(datapath+'sounding/'+f4+'.txt') # snow, 


ids, sids,  times = [], [], []
for f in [f1, f2, f3, f4]:
    id1, sid1, timetxt = f.split('_')
    time1 = timetxt.split('.')[0]
    time = time1[0:4]+'-'+time1[4:6]+'-'+time1[6:8]+' '+\
            time1[8:10]+':'+time1[10:12]+':'+time1[12:14]
    ids.append(id1)
    sids.append(sid1)
    times.append(time)

idx0 = rain[(rain.ID==ids[0]) & (rain.datetime==times[0])].index[0]
idx1 = snow[(snow.ID==ids[1]) & (snow.datetime==times[1])].index[0]
idx2 = rain[(rain.ID==ids[2]) & (rain.datetime==times[2])].index[0]
idx3 = snow[(snow.ID==ids[3]) & (snow.datetime==times[3])].index[0]
# ti -0.1 net small, net 0
# net 0 ti ---



ss = {
            'b': {'id':   ids[0], 
                  'time': times[0],
                  'ww':   '66\nFreezing rain (TR)',
                  'data': df1,
                  'me':   rain.loc[idx0, PA],
                  're':   rain.loc[idx0, NA],
                  'net':  rain.loc[idx0, 'net'],
                  'sp':   rain.loc[idx0, 'p'],
                  'ti':   rain.loc[idx0, 'ti']},
            
            'c': {'id':   ids[1],  # good
                  'time': times[1],
                  'ww':   '73\nSnow (TS)',
                  'data': df2,
                  'me':   snow.loc[idx1, PA],
                  're':   snow.loc[idx1, NA],
                  'net':  snow.loc[idx1, 'net'],
                  'sp':   snow.loc[idx1, 'p'],
                  'ti':   snow.loc[idx1, 'ti']},
            
            'd': {'id':   ids[2], 
                  'time': times[2],
                  'ww':   '67\nFreezing rain (TR)',
                  'data': df3,
                  'me':   rain.loc[idx2, PA],
                  're':   rain.loc[idx2, NA],
                  'net':  rain.loc[idx2, 'net'],
                  'sp':   rain.loc[idx2, 'p'],
                  'ti':   rain.loc[idx2, 'ti']},
            
            'e': {'id':   ids[3], 
                  'time': times[3],
                  'ww':   '71\nSnow (TS)',
                  'data': df4,
                  'me':   snow.loc[idx3, PA],
                  're':   snow.loc[idx3, NA],
                  'net':  snow.loc[idx3, 'net'],
                  'sp':   snow.loc[idx3, 'p'],
                  'ti':   snow.loc[idx3, 'ti']},
            }


coefs50 = coefs[0.5]

#%%
fig = plt.figure(figsize=(7, 8), dpi=1200)
ax0 = fig.add_axes([0.10, 0.55, 0.43, 0.35])
ax1 = fig.add_axes([0.66, 0.55, 0.28, 0.35])
ax2 = fig.add_axes([0.10, 0.08, 0.28, 0.35])
ax3 = fig.add_axes([0.38, 0.08, 0.28, 0.35])
ax4 = fig.add_axes([0.66, 0.08, 0.28, 0.35])
axes = [ax0, ax1, ax2, ax3, ax4]
labels = ['a', 'b', 'c', 'd', 'e']
for ax, label in zip(axes, labels):
    ax.set_title('('+label+')', loc='left')
    
# ax0 scatter
params = {'xlim': [-5, 5],
          'ylim': [-6, 1],
          'xlabel': 'ln('+str(npa)+'TiME/'+str(nna)+'TiRE)',
          'ylabel': 'Ti ('+chr(176)+'C)',
          'title': 'Type 2 Sounding'}
s1 = ax0.scatter(rain['net'], rain['ti'], color=blue, marker='o', facecolor='none', s=10, alpha=0.8)
s2 = ax0.scatter(snow['net'], snow['ti'], color=red, marker='^',facecolor='none', s=14, alpha=0.6)
ax0.plot([0, 0], [-8, 3], color='0.4')
ax0.plot([-10, 10], [0, 0],  color='0.4')
xxs = np.arange(-10, 10.1, 0.1)
ax0.plot(xxs, type2_tanh(xxs, *coefs50), color='k')
lg1 = ax0.legend([s1, s2],['rain', 'snow'])
ax0.set_ylim(params['ylim'])
ax0.set_ylabel(params['ylabel'])
ax0.set_xlim(params['xlim'])
ax0.set_xlabel(params['xlabel'])
ax0.set_title(params['title'])
ax0.text(0.5, -3.8, '50%', fontweight='bold') 
   
# plot the soundings
axes = [ax1, ax2, ax3, ax4]
dfs = [df1, df2, df3, df4]

for ax, label in zip(axes, labels[1:]):
    ax.scatter(ss[label]['ti'], -ss[label]['sp'], color='r', s=20,  marker=(5, 1))
    
    ax.plot(ss[label]['data']['ti'].values, 
            -ss[label]['data']['p'].values, marker='.')
    
    ax.plot([ss[label]['data']['ti'][0], 0], 
            [-ss[label]['data']['p'][0], -ss[label]['data']['p'][0]], 
            '--', color='k')
    ax.set_title('ww='+ss[label]['ww'], fontsize=10)
    ax.set_xlabel('Ti ('+chr(176)+'C)')

# settings for (b)
ax1.set_xlim(-5, 5)
ax1.set_ylim([-950, -750])
ax1.plot([0, 0], [-1010, -700], '--', color=red)
ax1.set_xticks([-5, -2, 0, 2, 5])
ax1.set_yticks(np.arange(-950, -700, 50))
ax1.set_yticklabels(-np.arange(-950, -700, 50))

# uniform settings for (c-e)
ax2.set_xlim(-8, 3)
ax3.set_xlim(-8, 4)
ax4.set_xlim(-8, 3)
# set yaxis
for ax in [ax2, ax3, ax4]:
    ax.set_xticks([-5, -2, 0, 2])
    
    ax.plot([0, 0], [-1010, -600], '--', color=red)
    
    ax.set_ylim([-1010, -750])
    ax.set_yticks(np.arange(-1000, -749, 50))
    ax.set_yticklabels(-np.arange(-1000, -749, 50))

 
# ------------------------------frame and grid settings
def remove_frame(ax2):
    '''remove left, top, right boundaries'''
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    return   
for ax in [ax1, ax2, ax3, ax4]:
    remove_frame(ax)
   
for ax in [ax1, ax2]:
    # recover the left frame and add label and grid line 
    ax.spines['left'].set_visible(True)
    ax.grid(axis='y')
    ax.set_ylabel('Pressure (hPa)')

def clean_y_axis(ax2):
    '''remove y axis labels and ticks, 
    but keep y grids same as the leftmost plot'''
    ax2.grid(axis='y')
    ax2.yaxis.set_ticklabels([])
    for tick in ax2.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
    return
for ax in [ax3, ax4]:
    clean_y_axis(ax)


pink = np.array((135, 0, 3))/255
labels = ['c', 'e']
for label in labels:
    net, tw = ss[label]['net'], ss[label]['ti']
    ax0.scatter(net, tw, color=pink, marker='*', s=130, zorder=5)
    ax0.text(net-0.4, tw+0.2, label,fontweight='bold', fontsize=12)
    
cyan = np.array((0, 24, 179))/255
labels = ['b', 'd']
for label in labels:
    net, tw = ss[label]['net'], ss[label]['ti']
    ax0.scatter(net, tw, color=cyan, marker='*', s=130, zorder=5)
    ax0.text(net-0.15, tw+0.3, label,fontweight='bold', fontsize=12)


# mark the energy
ax1.text(-1.5, -935, int(ss['b']['re']))
ax1.text(1,    -875, int(ss['b']['me']))
ax2.text(-2,   -925, int(ss['c']['re']))
ax2.text(0.7,  -835, int(ss['c']['me']))
ax3.text(-2,  -975, int(ss['d']['re']))
ax3.text(0.3,   -925, int(ss['d']['me']))
ax4.text(-2.3, -940, int(ss['e']['re']))
ax4.text(0.2, -890, int(ss['e']['me']))

fig.savefig('../figure/Figure8', bbox_inches='tight', dpi=1200)
fig.savefig('../figure/Figure8.pdf', format='pdf')




