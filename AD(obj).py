#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:40:54 2023

@author: yangziyuan
"""

#%%
import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all
df,dfrs,dfpt = biobase.biodata('all', 'all')


#%% summary
df,dfrs,dfpt = biobase.biodata('all', 'obj')

t = ['QFAMILY_2', 'QFAMILY_3', 'QFAMILY_5']
for i in df.index:
    for j in df.columns:
        if df[j][i] == 10:
            if j in t:
                df.loc[i,j] = 0
            else:
                df.loc[i,j] = np.nan
for i in dfrs.index:
    for j in dfrs.columns:
        if dfrs[j][i] == 10:
            if j in t:
                dfrs.loc[i,j] = 0
            else:
                dfrs.loc[i,j] = np.nan
for i in dfpt.index:
    for j in dfpt.columns:
        if dfpt[j][i] == 10:
            if j in t:
                dfpt.loc[i,j] = 0
            else:
                dfpt.loc[i,j] = np.nan
del i,j,t

# summary table
print(df.head())
describe = df.describe()
print(describe)
describe_rs = dfrs.describe()
describe_pt = dfpt.describe()
writer = pd.ExcelWriter('AD(obj)/describe.xlsx')
describe.to_excel(writer, sheet_name='All')
describe_rs.to_excel(writer, sheet_name='RS')
describe_pt.to_excel(writer, sheet_name='PT')
writer.save()
del describe,describe_rs,describe_pt,writer
biobase.resetCol('AD(obj)/describe.xlsx')


# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
t = -1
for i in ['GENDER', 'AGE', 
          'TIME_LIVING', 'EDU']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')
    ticks=[]
    for j in df.index:
        if df[i][j] not in ticks:
            ticks.append(df[i][j])
    ticks.sort()
    step = ticks[1] - ticks[0]
    #ticks.append(max(ticks)+step)
    #ticks.append(min(ticks)-step)
    ax.set_xticks(ticks)
    #ax.set_ylim(0, 1)
    del ticks,step,j,ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of Objective Variables - Personal', 
             x=0.46, y=0.92, fontsize=20)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-2
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
t = -1
for i in ['LICENSE', 'PRIVATE_CAR_1_1', 'PRIVATE_CAR_1_2']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':2}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')
    ticks=[]
    for j in df.index:
        if df[i][j] not in ticks:
            ticks.append(df[i][j])
    ticks.sort()
    step = ticks[1] - ticks[0]
    #ticks.append(max(ticks)+step)
    #ticks.append(min(ticks)-step)
    ax.set_xticks(ticks)
    #ax.set_ylim(0, 1)
    del ticks,step,j,ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of Objective Variables - Private Car', 
             x=0.45, fontsize=28)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-3
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['QHOUSEHOLD_1', 'QHOUSEHOLD_2', 'QHOUSEHOLD_3', 
          'QHOUSEHOLD_4', 'QHOUSEHOLD_5', 'FLOOR_AREA']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    if i != 'FLOOR_AREA':
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//3, t%3], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//3, t%3], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
        ticks=[]
        for j in df.index:
            if df[i][j] not in ticks:
                ticks.append(df[i][j])
        ticks.sort()
        step = ticks[1] - ticks[0]
        #ticks.append(max(ticks)+step)
        #ticks.append(min(ticks)-step)
        ax.set_xticks(ticks)
        #ax.set_ylim(0, 1)
        del ticks,step,j,ax
    else:
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//3, t%3], 
                        stat='density', binwidth=100, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//3, t%3], 
                        stat='density', binwidth=100, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
        del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of Objective Variables - House', 
             x=0.41, y=0.94, fontsize=28)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-4
fig, axs = plt.subplots(3, 3, figsize=(15, 13))
t = -1
for i in ['QFAMILY_1', 'QFAMILY_2', 'QFAMILY_3', 
          'QFAMILY_4', 'QFAMILY_5', 'QFAMILY_6', 
          'QFAMILY_7']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    if t != 6:
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//3, t%3], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//3, t%3], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
    if t == 6:
        plt.subplot(3,3,8)
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ax=sns.histplot(dfpt[i], color="red", label="PT", 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
    ticks=[]
    for j in df.index:
        if not np.isnan(df[i][j]):
            if df[i][j] not in ticks:
                ticks.append(df[i][j])
    ticks.sort()
    step = ticks[1] - ticks[0]
    #ticks.append(max(ticks)+step)
    #ticks.append(min(ticks)-step)
    ax.set_xticks(ticks)
    #ax.set_ylim(0, 1)
    del ticks,step,j,ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of Objective Variables - Family', 
             x=0.41, y=0.92, fontsize=28)
plt.delaxes(axs[2, 0])
plt.delaxes(axs[2, 2])
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-5
fig, axs = plt.subplots(2, 4, figsize=(20, 9))
t = -1
for i in ['WFH_ALLOWED', 'WFH_NOW', 'INCOME_1', 'INCOME_2', 
          'QOFFICE_1', 'QOFFICE_2', 'QOFFICE_3', 'QOFFICE_4']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    if i != 'INCOME_1' and i != 'INCOME_2':
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//4, t%4], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//4, t%4], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
        ticks=[]
        for j in df.index:
            if df[i][j] not in ticks:
                ticks.append(df[i][j])
        ticks.sort()
        step = ticks[1] - ticks[0]
        #ticks.append(max(ticks)+step)
        #ticks.append(min(ticks)-step)
        ax.set_xticks(ticks)
        #ax.set_ylim(0, 1)
        del ticks,step,j,ax
    else:
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//4, t%4], 
                        stat='density', binwidth=1000, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//4, t%4], 
                        stat='density', binwidth=1000, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
        del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of Objective Variables - Work', 
             x=0.34, y=0.94, fontsize=28)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-6
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['RS_EXPERIENCE_1', 'RS_EXPERIENCE_2', 'RS_EXPERIENCE_3', 
          'PT_EXPERIENCE_1', 'PT_EXPERIENCE_2', 'PT_EXPERIENCE_3']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    if t <= 2:
        ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//3, t%3], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='dodgerblue')
        ticks=[]
        for j in dfrs.index:
            if dfrs[i][j] not in ticks:
                ticks.append(dfrs[i][j])
        ticks.sort()
        step = ticks[1] - ticks[0]
        #ticks.append(max(ticks)+step)
        #ticks.append(min(ticks)-step)
        ax.set_xticks(ticks)
        #ax.set_ylim(0, 1)
        del ticks,step,j,ax
    else:
        ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//3, t%3], 
                        stat='density', discrete=True, shrink=.5, kde=True, 
                        kde_kws={'cut':1.8}, edgecolor='orangered')
        ticks=[]
        for j in dfpt.index:
            if dfpt[i][j] not in ticks:
                ticks.append(dfpt[i][j])
        ticks.sort()
        step = ticks[1] - ticks[0]
        #ticks.append(max(ticks)+step)
        #ticks.append(min(ticks)-step)
        ax.set_xticks(ticks)
        #ax.set_ylim(0, 1)
        del ticks,step,j,ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.subplot(2,3,3)
plt.legend()
plt.suptitle('Distribution Plots of Objective Variables - Experience', 
             x=0.44, y=0.94, fontsize=28)
plt.show()
del t,i,fig,axs


# 频率分布直方图2
biobase.scatterplot(df, 'QFAMILY_6', 'MODE_2023')


# Scatterplot Matrix
df,dfrs,dfpt = biobase.biodata('all', 'obj')
for i in df.index:
    for j in df.columns:
        if df[j][i] == 10:
            df[j][i] = 0
del i,j
# Create the plot
columns = list(df.columns)
columns.remove('NUM')
columns.remove('RS_AV')
columns.remove('PT_AV')
#columns.remove('MODE_2023')
g = sns.PairGrid(data=df, vars=columns, hue=None)
g.map_upper(biobase.reg_coef)
g = g.map_lower(sns.regplot, scatter_kws={"edgecolor": "white"})
g = g.map_diag(sns.histplot, kde=True)
plt.suptitle('Scatterplot Matrix of Subjective Variables', x=0.09, y=1, fontsize=46)
plt.show()
del columns,g


#%% modeling
df,dfrs,dfpt = biobase.biodata('all', 'obj')
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db

df.drop(columns=['RS_EXPERIENCE_1', 'RS_EXPERIENCE_2', 'RS_EXPERIENCE_3', 
                 'PT_EXPERIENCE_1', 'PT_EXPERIENCE_2', 'PT_EXPERIENCE_3'], 
        inplace = True)

t = ['QFAMILY_2', 'QFAMILY_3', 'QFAMILY_5']
for i in df.index:
    for j in df.columns:
        if df[j][i] == 10:
            if j in t:
                df.loc[i,j] = 0
del i,j,t

t = ['QFAMILY_4', 'QFAMILY_6', 'QFAMILY_7']
for i in df.index:
    for j in t:
        if df[j][i] == 10:
            df.loc[i,j] = 0
        else:
            df.loc[i,j] = df[j][i] + 1
del i,j,t

for i in df.columns:
    globals()[i] = Variable(i) 
del i

database = db.Database('RSPT_objective', df)
# Parameters to be estimated
ASC_RS = Beta('ASC_RS', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)

# personal
B_GENDER = Beta('B_GENDER', 0, None, None, 0)
B_AGE = Beta('B_AGE', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_EDU = Beta('B_EDU', 0, None, None, 0)

# car
B_LICENSE = Beta('B_LICENSE', 0, None, None, 0)
B_CAR1 = Beta('B_CAR1', 0, None, None, 0)
B_CAR2 = Beta('B_CAR2', 0, None, None, 0)

# house
B_H1 = Beta('B_H1', 0, None, None, 0)
B_H2 = Beta('B_H2', 0, None, None, 0)
B_H3 = Beta('B_H3', 0, None, None, 0)
B_H4 = Beta('B_H4', 0, None, None, 0)
B_H5 = Beta('B_H5', 0, None, None, 0)
B_FA = Beta('B_FA', 0, None, None, 0)

# family
B_F1 = Beta('B_F1', 0, None, None, 0)
B_F2 = Beta('B_F2', 0, None, None, 0)
B_F3 = Beta('B_F3', 0, None, None, 0)
B_F4 = Beta('B_F4', 0, None, None, 0)
B_F5 = Beta('B_F5', 0, None, None, 0)
B_F6 = Beta('B_F6', 0, None, None, 0)
B_F7 = Beta('B_F7', 0, None, None, 0)

# work
B_WFHA = Beta('B_WFHA', 0, None, None, 0)
B_WFHN = Beta('B_WFHN', 0, None, None, 0)
B_I1 = Beta('B_I1', 0, None, None, 0)
B_I2 = Beta('B_I2', 0, None, None, 0)
B_O1 = Beta('B_O1', 0, None, None, 0)
B_O2 = Beta('B_O2', 0, None, None, 0)
B_O3 = Beta('B_O3', 0, None, None, 0)
B_O4 = Beta('B_O4', 0, None, None, 0)


# Scaleing variables
GENDER = GENDER / 100
AGE = AGE / 100
TIME_LIVING = TIME_LIVING / 100000
EDU = EDU / 100

PRIVATE_CAR_1_1 = PRIVATE_CAR_1_1 / 10
PRIVATE_CAR_1_2 = PRIVATE_CAR_1_2 / 10

QHOUSEHOLD_1 = QHOUSEHOLD_1 / 1
QHOUSEHOLD_2 = QHOUSEHOLD_2 / 10
QHOUSEHOLD_3 = QHOUSEHOLD_3 / 10
QHOUSEHOLD_4 = QHOUSEHOLD_4 / 10
QHOUSEHOLD_5 = QHOUSEHOLD_5 / 10
FLOOR_AREA = FLOOR_AREA / 10000

QFAMILY_1 = QFAMILY_1 / 10
QFAMILY_2 = QFAMILY_2 / 10
QFAMILY_3 = QFAMILY_3 / 10
QFAMILY_4 = QFAMILY_4 / 10
QFAMILY_5 = QFAMILY_5 / 10
QFAMILY_6 = QFAMILY_6 / 10
QFAMILY_7 = QFAMILY_7 / 100

WFH_ALLOWED = WFH_ALLOWED / 10
WFH_NOW = WFH_NOW / 10
INCOME_1 = INCOME_1 / 10000
INCOME_2 = INCOME_2 / 10000
QOFFICE_1 = QOFFICE_1 / 10
QOFFICE_2 = QOFFICE_2 / 10
QOFFICE_3 = QOFFICE_3 / 100
QOFFICE_4 = QOFFICE_4 / 100


# Definition of the utility functions
'''V1 = (ASC_RS + B_GENDER*GENDER + B_AGE*AGE + B_TIME*TIME_LIVING + B_EDU*EDU + 
      
      B_LICENSE*LICENSE + B_CAR1*PRIVATE_CAR_1_1 + B_CAR2*PRIVATE_CAR_1_2 + 
      
      B_H1*QHOUSEHOLD_1 + B_H2*QHOUSEHOLD_2 + B_H3*QHOUSEHOLD_3 + 
      B_H4*QHOUSEHOLD_4 + B_H5*QHOUSEHOLD_5 + B_FA*FLOOR_AREA + 
      
      B_F1*QFAMILY_1 + B_F2*QFAMILY_2 + B_F3*QFAMILY_3 + B_F4*QFAMILY_4 + 
      B_F5*QFAMILY_5 + B_F6*QFAMILY_6 + B_F7*QFAMILY_7 + 
      
      B_WFHA*WFH_ALLOWED + B_WFHN*WFH_NOW + B_I1*INCOME_1 + B_I2*INCOME_2 + 
      B_O1*QOFFICE_1 + B_O2*QOFFICE_2 + B_O3*QOFFICE_3 + B_O4*QOFFICE_4
      )'''

V1 = (ASC_RS + B_AGE*AGE + 
      
      B_LICENSE*LICENSE + B_CAR1*PRIVATE_CAR_1_1 + 
      
      B_H1*QHOUSEHOLD_1 + B_H3*QHOUSEHOLD_3 + 
      B_H4*QHOUSEHOLD_4 + B_H5*QHOUSEHOLD_5 + B_FA*FLOOR_AREA + 
      
      B_F1*QFAMILY_1 + B_F3*QFAMILY_3 + B_F4*QFAMILY_4 + 
      B_F6*QFAMILY_6 + 
      
      B_WFHA*WFH_ALLOWED + B_WFHN*WFH_NOW + B_I2*INCOME_2
      )

V2 = ASC_PT


# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2}

# Associate the availability conditions with the alternatives
av = {1: RS_AV, 2: PT_AV}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, MODE_2023)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'default_objective' #change the model name here

# Estimate the parameters
the_biogeme.generate_html = False
the_biogeme.generate_pickle = False
the_biogeme.save_iterations = False
results = the_biogeme.estimate()
print(results.short_summary())

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)

# Get the results in Html
biobase.newwriteHtml(results,
    path='AD(obj)/'
)



