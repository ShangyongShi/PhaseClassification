# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:01:07 2023

sounding data for figure 2

@author: ssynj
"""
import pandas as pd
from function.watervapor import rh2tw

path = '../02-Output/sounding/'
df_t = pd.read_csv(path +'CAM00071625_71625_temp_sounding.txt')
df_t.set_index(pd.to_datetime(df_t.loc[:, 'year':'hour']), inplace=True)

df_t2 = pd.read_csv(path +'USM00072203_72203_temp_sounding.txt')
df_t2.set_index(pd.to_datetime(df_t2.loc[:, 'year':'hour']), inplace=True)

df_t3 = pd.read_csv(path +'USM00072208_72208_temp_sounding.txt')
df_t3.set_index(pd.to_datetime(df_t3.loc[:, 'year':'hour']), inplace=True)

df_t3 = pd.read_csv(path +'USM00070231_70231_temp_sounding.txt')
df_t3.set_index(pd.to_datetime(df_t3.loc[:, 'year':'hour']), inplace=True)
df_rh3 = pd.read_csv(path +'USM00070231_70231_rh_sounding.txt')
df_rh3.set_index(pd.to_datetime(df_rh3.loc[:, 'year':'hour']), inplace=True)

# TYPE 0A ALL COLD 71625
time = '2005-11-17 15:00:00'
time = '2005-11-15 14:00:00'
time = '1987-03-04 12:00:00'
p01 = df_t.loc[time, '0':'10'].map(lambda x:x.split(',')[0]).astype(float)
t01 = df_t.loc[time, '0':'10'].map(lambda x:x.split(',')[1]).astype(float)
t01 = t01[~t01.isna() & ~p01.isna()]
p01 = p01[~t01.isna() & ~p01.isna()]

#TYPE0B 72203 all warm
time = '1978-11-08 00:00:00'
p02 = df_t2.loc[time, '0':'10'].map(lambda x:x.split(',')[0]).astype(float)
t02 = df_t2.loc[time, '0':'10'].map(lambda x:x.split(',')[1]).astype(float)
t02 = t02[~t02.isna() & ~p02.isna()]
p02 = p02[~t02.isna() & ~p02.isna()]

# TYPE1  WARM LAYER 71625 1988 10 11 12
time = '1988-10-11 12:00:00'
p1 = df_t.loc[time, '0':'10'].map(lambda x:x.split(',')[0]).astype(float)
t1 = df_t.loc[time, '0':'10'].map(lambda x:x.split(',')[1]).astype(float)
t1 = t1[~t1.isna() & ~p1.isna()]
p1 = p1[~t1.isna() & ~p1.isna()]

#TYPE 2 MELT AND REFREEZING
time = '1978-02-09 12:00:00'
time = '1986-11-02 12:00:00'
# time = '1986-11-03 12:00:00'
p2 = df_t3.loc[time, '0':'10'].map(lambda x:x.split(',')[0]).astype(float)
t2 = df_t3.loc[time, '0':'10'].map(lambda x:x.split(',')[1]).astype(float)
t2 = t2[~t2.isna() & ~p2.isna()]
p2 = p2[~t2.isna() & ~p2.isna()]
rh2 = df_rh3.loc[time, '0':'10'].map(lambda x:x.split(',')[1]).astype(float)
rh2 = rh2.loc[t2.index]
tw2 = t2.copy()
for i in range(len(t2)):
    tw2[i] =rh2tw(p2[i], t2[i], rh2[i])
tw2[5] = 2.9

sounding0A = pd.DataFrame(columns=['p', 't'])
sounding0B = pd.DataFrame(columns=['p', 't'])
sounding1 = pd.DataFrame(columns=['p', 't'])
sounding2 = pd.DataFrame(columns=['p', 't'])

sounding0A['p'] = p01
sounding0A['t'] = t01
sounding0B['p'] = p02
sounding0B['t'] = t02
sounding1['p'] = p1
sounding1['t'] = t1
sounding2['p'] = p2
sounding2['t'] = t2
sounding0A.to_csv('sounding0A.txt', index=False)
sounding0B.to_csv('sounding0B.txt', index=False)
sounding1.to_csv('sounding1.txt', index=False)
sounding2.to_csv('sounding2.txt', index=False)