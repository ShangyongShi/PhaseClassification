#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:34:59 2024

@author: ssynj
"""

import pandas as pd
import numpy as np
from scipy import optimize as opt

import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score

import sympy as sp
from sympy import solve, symbols
from scipy.optimize import curve_fit
from skimage import measure

from function.evaluate import *
from function.type1 import *
from function.conp import *
blue = np.array([14, 126, 191])/255
red = np.array([235, 57, 25])/255
 

datapath = '../../../data/NA/revision_t_tw_ti/'
datapath = '../data/'
figpath = '../figure/'

'''
Tw + TwME
'''
df = pd.read_csv(datapath+'final_all_cleaned_data.txt', index_col=0, dtype={'ID':str})

type1_ti_de = pd.read_csv(datapath+'type1_ti_de.txt', 
                          dtype={'ID':str, 'NA':float},index_col=0)
type1_ti_de.rename(columns={'PA_ti':'PA', 'NA_ti':'NA'},
                   inplace=True)

rain = type1_ti_de[(type1_ti_de.wwflag==1)]
snow = type1_ti_de[(type1_ti_de.wwflag==2)]


params = [0, 2.5, 0.5,
           0, 8, 1]
conp, num_rain, num_snow = conp_2d(rain, snow, 'ti', 'PA', params)
conp1, num_rain, num_snow = running_conp(rain, snow, 'ti', 'PA', params)
conp2 = running_mean(conp1, 0.5)

   
from sklearn.metrics import r2_score
from function.fitting import get_contour_values
[xs, ys] = get_contour_values(conp2)

# fit contour with desired function
def type1_exp(x, m, t):
    return m * np.exp(-t* x )
def type1_linear(x, m, t):
    return m * x +t

model = type1_exp
coefs = {}
initial_guess = [ 1.0, 0.2]
for value in np.arange(0.2, 0.81, 0.1):
    key = np.round(value, decimals=1)
    X, Y = xs[key], ys[key]
    coef, _ = curve_fit(model, X, Y, p0=initial_guess)
    
    print(key, r2_score(Y, model(X, *coef)))
    coefs[key] = coef
# =============================================================================
#    
# key = 0.7
# [xs7, ys7] = get_contour_values(conp)
# X, Y = xs7[key], ys7[key]
# X = np.concatenate([X, np.array([2, 4])])
# Y = np.pad(Y, [0, 2], 'constant')
# coef, _ = curve_fit(model, X, Y, p0=[0.7, 0.5])
# print(key, r2_score(Y, model(X, *coef)))
# coefs[0.7] = coef
# 
# =============================================================================
xxs = np.arange(0, 10, 0.01)
fig, ax = plt.subplots()
for value in np.arange(0.2, 0.71, 0.1):
    key = np.round(value, decimals=1)
    ax.plot(xxs, type1_exp(xxs, *coefs[key]))


# running mean figure
fig = plt.figure(figsize=(5, 4), dpi=300)
ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

# shadings
pc = ax.contourf(conp2.index, conp2.columns, conp2.T, 
                 cmap='jet', 
                 origin='lower', levels=np.arange(0, 1.1, 0.1))
# contours
ct = ax.contour(conp2.index, conp2.columns, conp2.T, 
                colors='k', linewidths=0.5,
                origin='lower', levels=np.arange(0, 1.1, 0.1))
ax.clabel(ct, ct.levels, inline=True, fmt='%3.1f', fontsize=10)
ax.set_xlabel('TiME (J/kg)')
ax.set_ylabel('Near surface Ti ('+chr(176)+'C)')
ax.set_ylim([0, 2.3])
ax.set_xticks(np.arange(0, 8, 1))
ax.set_title('Running Mean Conditional Probability')
cb = plt.colorbar(pc, cax=fig.add_axes([0.9, 0.15, 0.03, 0.7]), 
                  label='Conditional Probability of Solid Precipitation')
plt.savefig(figpath+'type1_running_contour_ti',  dpi=300, bbox_inches='tight')

# --------------------------------------------------
# sup figure 5

# =============================================================================
# fig = plt.figure(figsize=(5, 4), dpi=300)
# ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
# 
# # shadings
# pc = ax.contourf(conp2.index, conp2.columns, conp2.T, 
#                  cmap='jet', 
#                  origin='lower', levels=np.arange(0, 1.1, 0.1))
# # contours
# ct = ax.contour(conp2.index, conp2.columns, conp2.T, 
#                 colors='k', linewidths=0.5,
#                 origin='lower', levels=np.arange(0, 1.1, 0.1))
# ax.clabel(ct, ct.levels, inline=True, fmt='%3.1f', fontsize=10)
#     
# xxs = np.arange(0, 8.1, 0.1)
# for value in np.arange(0.3, 0.71, 0.1):
#     key = np.round(value, decimals=1)
#     coef = coefs[key]
#     ax.plot(xxs, model(xxs, *coef), '--', color='tab:red')
#   
# ax.set_xlim([0.25, 7.5])
# cb = plt.colorbar(pc, cax=fig.add_axes([0.9, 0.15, 0.03, 0.7]), 
#                   label='Conditional Probability of Solid Precipitation')
# ax.set_xlabel('TiME (J/kg)')
# ax.set_ylabel('Near surface Ti ('+chr(176)+'C)')
# ax.set_ylim([0.25, 2.25])
# ax.set_xticks(np.arange(1, 8, 1))
# ax.set_title('Type 1 Contours for Fitting')
# plt.savefig(figpath+'type1_contourfit_ti',  dpi=300, bbox_inches='tight')
# plt.savefig(figpath+'FigureS1',  dpi=300, bbox_inches='tight')
# 
# =============================================================================

# ---------------- figure 5
# =============================================================================
# xmin=0
# ymin=0
# xmax = 15
# ymax = 3
# tiPA = 1.838190222582581
# 
# fig = plt.figure(figsize=(6, 5), dpi=300)
# ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
# #s2 = ax.scatter(snow['PA'], snow.tw, color=red, marker='^',facecolor='none', s=14, alpha=0.7)
# 
# s1 = ax.scatter(rain['PA'], rain.ti, color=blue, marker='o', facecolor='none', s=10, alpha=0.7)
# s2 = ax.scatter(snow['PA'], snow.ti, color=red, marker='^',facecolor='none', s=14, alpha=0.7)
# 
# ax.plot([tiPA, tiPA], [0, 3], '--', color='0.3')
# 
# x = np.arange(0, xmax+0.01, 0.01)
# ax.plot(x, type1_exp(x, *coefs[0.3]), 'k', linestyle=(0, (1, 1))) # dotted
# ax.plot(x, type1_exp(x, *coefs[0.5]), 'k') # solid
# ax.plot(x, type1_exp(x, *coefs[0.7]), 'k', linestyle=(0, (5, 1))) # dashed
# ax.text(1, 0.27, '70%', fontweight='bold')
# ax.text(6.5, 0.23, '50%', fontweight='bold')
# ax.text(10, 0.33, '30%', fontweight='bold')
# lg = ax.legend([s1, s2], ['rain', 'snow'])
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([xmin, ymax])
# ax.set_xlabel('TiME (J/kg)', fontsize=12)
# ax.set_ylabel('Near-surface Ti ('+ chr(176) + 'C)', fontsize=12)
# ax.tick_params(axis='x', labelsize=10)
# ax.tick_params(axis='y', labelsize=10)
# ax.set_title('Type 1 Sounding\n Separation Lines Between Rain and Snow', fontsize=12)
# 
# 
# =============================================================================

# %%---------------- figure 5
xmin=0
ymin=0
xmax = 15
ymax = 3
tiPA = 1.838190222582581

cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(21*cm, 9*cm), dpi=1200)

ax = fig.add_axes([0.07, 0.15, 0.35, 0.68])
ax2 = fig.add_axes([0.53, 0.15, 0.35, 0.68])
cbax = fig.add_axes([0.91, 0.15, 0.01, 0.68])

s1 = ax.scatter(rain['PA'], rain.ti, color=blue, marker='o', facecolor='none',linewidths=0.5, s=8, alpha=0.7)
s2 = ax.scatter(snow['PA'], snow.ti, color=red, marker='^',facecolor='none',linewidths=0.5, s=12, alpha=0.7)

ax.plot([tiPA, tiPA], [0, 3], '--', color='0.3', linewidth=2.5)

x = np.arange(0, xmax+0.01, 0.01)
ax.plot(x, type1_exp(x, *coefs[0.3]), 'k', linestyle=(0, (1, 1))) # dotted
ax.plot(x, type1_exp(x, *coefs[0.5]), 'k') # solid
ax.plot(x, type1_exp(x, *coefs[0.7]), 'k', linestyle=(0, (5, 1))) # dashed
ax.text(1, 0.27, '70%', fontweight='bold')
ax.text(6.5, 0.23, '50%', fontweight='bold')
ax.text(10, 0.33, '30%', fontweight='bold')
lg = ax.legend([s1, s2], ['rain', 'snow'])
ax.set_xlim([xmin, xmax])
ax.set_ylim([xmin, ymax])
ax.set_xlabel('TiME (J/kg)', fontsize=12)
ax.set_ylabel('Near-surface Ti ('+ chr(176) + 'C)', fontsize=12)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)





# shadings
pc = ax2.contourf(conp2.index, conp2.columns, conp2.T, 
                 cmap='jet', 
                 origin='lower', levels=np.arange(0, 1.1, 0.1))
# contours
ct = ax2.contour(conp2.index, conp2.columns, conp2.T, 
                colors='k', linewidths=0.5,
                origin='lower', levels=np.arange(0, 1.1, 0.1))
ax2.clabel(ct, ct.levels, inline=True, fmt='%3.1f', fontsize=10)
    
xxs = np.arange(0, 8.1, 0.1)
for value in np.arange(0.3, 0.71, 0.1):
    key = np.round(value, decimals=1)
    coef = coefs[key]
    ax2.plot(xxs, model(xxs, *coef), '--', color='tab:red')
  
ax2.set_xlim([0.25, 7.5])
cb = plt.colorbar(pc, cax=cbax, 
                  label='Conditional Probability of Snow')
ax2.set_xlabel('TiME (J/kg)')
ax2.set_ylabel('Near surface Ti ('+chr(176)+'C)')
ax2.set_ylim([0.25, 2.25])
ax2.set_xticks(np.arange(1, 8, 1))

ax.set_title('(a)', loc='left', fontsize=12)
ax2.set_title('(b)', loc='left', fontsize=12)
fig.suptitle('Type 1 Soundings', x=0.52)


plt.savefig(figpath+'Figure5.pdf', format='pdf')
plt.savefig(figpath+'Figure5', dpi=1200, bbox_inches='tight')


# evaluate
#%%
type1_ti_ev = pd.read_csv(datapath+'type1_ti_ev.txt', index_col=0)
type1_ti_ev.rename(columns={'PA_ti':'PA', 'NA_ti':'NA'}, inplace=True)

print('--- evaluate model type1 exp tw')
_=evaluate_model(type1_ti_ev, 'PA', 'tw', type1_exp, coefs[0.5])

print('--- evaluate model type1 exp ti')
_=evaluate_model(type1_ti_ev, 'PA', 'ti', type1_exp, coefs[0.5])
_=evaluate_model(type1_ti_ev, 'PA', 'ti', type1_exp, coefs[0.7])

evaluate_probsnow(type1_ti_ev, 0.5, 'tw')


