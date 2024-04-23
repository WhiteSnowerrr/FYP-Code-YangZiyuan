#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:44:01 2023

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
df,dfrs,dfpt = biobase.biodata('all', 'subj')

# summary table
print(df.head())
describe = df.describe()
print(describe)
describe_rs = dfrs.describe()
describe_pt = dfpt.describe()
writer = pd.ExcelWriter('BE(subj)/describe.xlsx')
describe.to_excel(writer, sheet_name='All')
describe_rs.to_excel(writer, sheet_name='RS')
describe_pt.to_excel(writer, sheet_name='PT')
writer.save()
del describe,describe_rs,describe_pt,writer
biobase.resetCol('BE(subj)/describe.xlsx')


# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['CS_RS_1_1', 'CS_RS_1_2', 'ATT_PT_1', 'ATT_PT_2', 'ATT_PT_3', 'ATT_PT_4']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
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
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of Subjective Variables', 
             x=0.37, y=0.94, fontsize=28)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-2
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
t = -1
for i in ['ATT_SHIFT_RS_1', 'ATT_SHIFT_RS_2', 'ATT_SHIFT_RS_3']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ticks=[]
    for j in dfrs.index:
        if dfrs[i][j] not in ticks:
            ticks.append(df[i][j])
    ticks.sort()
    step = ticks[1] - ticks[0]
    #ticks.append(max(ticks)+step)
    #ticks.append(min(ticks)-step)
    ax.set_xticks(ticks)
    ax.set_ylim(0, 1.5)
    del ticks,step,j,ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend() 
plt.suptitle('Distribution Plots of RS Shift Variables', 
             x=0.35, fontsize=28)
plt.show()
del t,i,fig,axs


# 频率分布直方图2
biobase.scatterplot(df, 'ATT_PT_2', 'MODE_2023')


# Scatterplot Matrix
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
plt.suptitle('Scatterplot Matrix of Subjective Variables', ha='right', fontsize=42)
plt.show()
del columns,g


#%% modeling
df,dfrs,dfpt = biobase.biodata('all', 'subj')
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db

for i in df.columns:
    globals()[i] = Variable(i) 
del i

database = db.Database('RSPT_subjective', df)
# Parameters to be estimated
ASC_RS = Beta('ASC_RS', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
'''
B_ATT1_RS = Beta('B_ATT1_RS', 0, None, None, 0)
B_ATT2_RS = Beta('B_ATT2_RS', 0, None, None, 0)
B_ATT3_RS = Beta('B_ATT3_RS', 0, None, None, 0)
B_ATT4_RS = Beta('B_ATT4_RS', 0, None, None, 0)
B_CSRS1_RS = Beta('B_CSRS1_RS', 0, None, None, 0)
B_CSRS2_RS = Beta('B_CSRS2_RS', 0, None, None, 0)'''

B_ATT1 = Beta('B_ATT1', 0, None, None, 0)
B_ATT2 = Beta('B_ATT2', 0, None, None, 0)
B_ATT3 = Beta('B_ATT3', 0, None, None, 0)
B_ATT4 = Beta('B_ATT4', 0, None, None, 0)
B_CSRS1 = Beta('B_CSRS1', 0, None, None, 0)
B_CSRS2 = Beta('B_CSRS2', 0, None, None, 0)


# Scaleing variables
ATT_PT_1 = ATT_PT_1 / 10
ATT_PT_2 = ATT_PT_2 / 10
ATT_PT_3 = ATT_PT_3 / 10
ATT_PT_4 = ATT_PT_4 / 10
CSRS_1_1 = CSRS_1_1 / 10
CSRS_1_2 = CSRS_1_2 / 10

# Definition of the utility functions
V1 = (ASC_RS + B_ATT1 * ATT_PT_1 + B_ATT2 * ATT_PT_2 + 
      B_ATT3 * ATT_PT_3 + B_ATT4 * ATT_PT_4 + 
      B_CSRS1 * CSRS_1_1 + B_CSRS2 * CSRS_1_2)

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
the_biogeme.modelName = 'default_subjective' #change the model name here

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
    path='BE(subj)/'
)








