#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4

@author: ssynj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from function.evaluate import print_metrics
# from function.type1 import *
# from function.conp import *
def ConpExp(x, m, t):
    '''exponential function'''
    return m * np.exp(-t * x)+1-m

def conp_PA(df):
    df_rain = df[df.wwflag==1]
    df_snow = df[df.wwflag==2]

    pre_rain = df_rain.loc[:, 'PA']
    pre_snow = df_snow.loc[:, 'PA']

    binsize=2
    bins = np.arange(binsize/2, 200, binsize)
    num_rain = pd.DataFrame(data=0, index=[0], columns=bins)
    num_snow = pd.DataFrame(data=0, index=[0], columns=bins)
    for col in num_rain.columns:
        num_rain.loc[0, col] += ((pre_rain>col-binsize/2) & (pre_rain<=col+binsize/2)).sum()
        num_snow.loc[0, col] += ((pre_snow>col-binsize/2) & (pre_snow<=col+binsize/2)).sum()
    conp = num_snow/(num_snow+num_rain)

    xs, ys = conp.columns.values, conp.loc[0]
    popt, pcov = curve_fit(ConpExp, xs, ys)
    y_pred = ConpExp(xs, *popt)
    r2_score(ys, y_pred)

    x50 = -np.log((0.5+popt[0]-1)/popt[0])/popt[1]

    return conp, x50, popt 

 

datapath = '../data/'
figpath = '../figure/'


compare_type1 = pd.read_csv(datapath+'compare_type1_training.txt', 
                            dtype={'ID':str}, index_col=0)
compare_type1_t_de = compare_type1.copy()
compare_type1_t_de.rename(columns={'PA_t':'PA', 'NA_t':'NA'}, inplace=True)

compare_type1_tw_de = compare_type1.copy()
compare_type1_tw_de.rename(columns={'PA_tw':'PA', 'NA_tw':'NA'}, inplace=True)

compare_type1_ti_de = compare_type1.copy()
compare_type1_ti_de.rename(columns={'PA_ti':'PA', 'NA_ti':'NA'}, inplace=True)

conp_t,  tPA, popt_t  = conp_PA(compare_type1_t_de)
conp_tw, twPA, popt_tw  = conp_PA(compare_type1_tw_de)
conp_ti, tiPA, popt_ti = conp_PA(compare_type1_ti_de)

print(tPA, twPA, tiPA)

test = pd.read_csv(datapath+'compare_type1_evaluation.txt', index_col=0)
print('compare type 1 test data size:', len(test))
# t PA threshold
pre_rain = test[test.PA_t>tPA]['wwflag']
pre_snow = test[test.PA_t<=tPA]['wwflag']
print('\n---T, ME')
_ = print_metrics(pre_rain, pre_snow, True)

# tw PA threshold
pre_rain = test[test.PA_tw>twPA]['wwflag']
pre_snow = test[test.PA_tw<=twPA]['wwflag']
print('\n--- Tw, ME')
_ = print_metrics(pre_rain, pre_snow, True)

# ti PA threshold
pre_rain = test[test.PA_ti>tiPA]['wwflag']
pre_snow = test[test.PA_ti<=tiPA]['wwflag']
print('\n--- Ti, ME')
_ = print_metrics(pre_rain, pre_snow, True)


def plot_conp_PA(ax, conp, x50, popt, params):
    l1, = ax.plot(conp.columns.values, conp.loc[0].values, 'r.-')
    xfit = np.arange(0, 150, 0.1)
    l2, = ax.plot(xfit, ConpExp(xfit, *popt), 'k-', alpha=0.8)
    ax.plot([0, 1000], [0.5, 0.5], 'k--', alpha=0.6)
    l3, = ax.plot([x50, x50], [0, 1], 'k--', alpha=0.6)
    ax.set_xticks(np.arange(0, 60, 10))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1])
    ax.grid()
    #ax.legend([l1, l2, l3], ['Obs', 'Fit', '{:3.1f}'.format(x50)+' J kg$^{-1}$'])
    ax.set_xlabel('Melting Energy (J/Kg)')
    #ax.set_ylabel('Conditional Probability of Snow')
    ax.set_title(params['title'])
    return l1, l2, l3
#%%
cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(19*cm, 8*cm), dpi=1200)
ax1 = fig.add_axes([0.10, 0.15, 0.24, 0.6])
ax2 = fig.add_axes([0.42, 0.15, 0.24, 0.6])
ax3 = fig.add_axes([0.74, 0.15, 0.24, 0.6])

plot_conp_PA(ax1, conp_t,  tPA, popt_t, {'tstr':'T', 'title':'T'})
plot_conp_PA(ax2, conp_tw, twPA, popt_tw, {'tstr':'Tw', 'title':'Tw'})
l1, l2, l3 = plot_conp_PA(ax3, conp_ti, tiPA, popt_ti, {'tstr':'Ti', 'title':'Ti'})
ax1.set_ylabel('Conditional Probability of Snow')

ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')

ax2.legend([l1, l2], ['Obs', 'Fit'], loc='upper center')

for ax, varname, t50 in zip([ax1, ax2, ax3], ['TME', 'TwME', 'TiME'], [tPA, twPA, tiPA]):
    ax.text(7.5, 0.535, varname+'$_{50\%}$'+'={:3.1f}'.format(t50)+' J kg$^{-1}$')
fig.suptitle('Type 1 Soundings', y=0.93)
plt.savefig(figpath+'Figure4', bbox_inches='tight', dpi=1200)
plt.savefig(figpath+'Figure4.pdf',format='pdf')