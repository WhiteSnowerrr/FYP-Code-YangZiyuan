#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:34:29 2023

@author: yangziyuan
"""

import biobase
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
def reg_coef(x,y,label=None,color=None, **kwargs):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    if p < 0.01:
        sig_level = '***'
    elif p < 0.05:
        sig_level = '**'
    elif p < 0.05:
        sig_level = '*'
    else:
        sig_level = ''
        
    ax.annotate('r = {:.2f} {}'.format(r, sig_level), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.texts[0].set_size(16)
    ax.set_axis_off()

def scatterplot(df,x,y):
    import random
    random.seed(20020906)
    dftem = df.copy()
    for i in dftem.index:
        for j in df.columns:
            t = 10**len(str(dftem[j][i]))*0.0001
            dftem.loc[i,j] = dftem[j][i] + random.uniform(-t, t)
    del i,j,t

    plt.plot( x, y, 
             data=dftem, linestyle='', marker='o', 
             markersize=3, alpha=0.05, color="red")
    plt.xlabel(x) # 设置x轴标签
    plt.ylabel(y) # 设置y轴标签
    plt.title('Scatterplot of ' + x +' & '+ y, loc='left') # 设置标题




def cDFP(Data,vname=''):
    name=Data.name
    Fre_df = biobase.cDF(Data)
    plot=plt.figure()
    ax1=plot.add_subplot(1,1,1)
    ax1.plot(Fre_df['Rds'],Fre_df['cumsum'],color="skyblue")
    ax1.set_title(vname+" CDF")
    ax1.set_xlabel(name)
    ax1.set_ylabel("Fn("+str(name)+')')
    plt.show()