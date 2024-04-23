#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:25:33 2023

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
df,dfrs,dfpt = biobase.biodata('all', 'sp')

for i in dfrs.index:
    if dfrs['CS_1'][i] == 4:
        dfrs.loc[i,'CS_1'] = 3
    if dfrs['CS_2'][i] == 4:
        dfrs.loc[i,'CS_2'] = 3
for i in dfpt.index:
    if dfpt['CS_1'][i] == 4:
        dfpt.loc[i,'CS_1'] = 3
    if dfpt['CS_2'][i] == 4:
        dfpt.loc[i,'CS_2'] = 3
del i

# summary table
dfrs_rh = dfrs.copy()
dfrs_rp = dfrs.copy()
for i in dfrs.index:
    if dfrs['RS3_SHARE'][i] == 0:
        dfrs_rp.drop(i, inplace = True)
    else:
        dfrs_rh.drop(i, inplace = True)
dfpt_rh = dfpt.copy()
dfpt_rp = dfpt.copy()
for i in dfpt.index:
    if dfpt['RS3_SHARE'][i] == 0:
        dfpt_rp.drop(i, inplace = True)
    else:
        dfpt_rh.drop(i, inplace = True)
print(df.head())
describe = df.describe()
print(describe)
describe_rsrh = dfrs_rh.describe()
describe_rsrp = dfrs_rp.describe()
describe_ptrh = dfpt_rh.describe()
describe_ptrp = dfpt_rp.describe()
writer = pd.ExcelWriter('CF(sp)/describe.xlsx')
describe.to_excel(writer, sheet_name='All')
describe_rsrh.to_excel(writer, sheet_name='RS_RH')
describe_ptrh.to_excel(writer, sheet_name='PT_RH')
describe_rsrp.to_excel(writer, sheet_name='RS_RP')
describe_ptrp.to_excel(writer, sheet_name='PT_RP')
writer.save()
del describe,describe_rsrh,describe_rsrp,describe_ptrh,describe_ptrp,writer
del i,dfrs_rh,dfrs_rp,dfpt_rh,dfpt_rp
biobase.resetCol('CF(sp)/describe.xlsx')


# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['PT1_WAIT', 'PT1_WALK', 'PT1_TT', 
          'PT1_COST', 'PT1_TRANS', 'PT1_CROWD']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//3, t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//3, t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')
    if t != 1 and t != 2 and t != 3:
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
        del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - PT', 
             x=0.34, y=0.94, fontsize=28)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-2
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
t = -1
for i in ['CS2_WALK', 'CS2_TT', 
          'CS2_COST', 'CS2_DISIN']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')
    if t != 1 and t != 2:
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
        del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - CS', 
             x=0.36, y=0.92, fontsize=20)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-3
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
t = -1
for i in ['RS3_WAIT', 'RS3_TT', 
          'RS3_COST', 'RS3_SHARE']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs[i], color="skyblue", label="RS", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt[i], color="red", label="PT", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')
    if t != 1 and t != 2:
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
        del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - RS', 
             x=0.34, y=0.92, fontsize=20)
plt.show()
del t,i,fig,axs

# 频率分布直方图 1-4
dfrs_rh = dfrs.copy()
dfrs_rp = dfrs.copy()
for i in dfrs.index:
    if dfrs['RS3_SHARE'][i] == 0:
        dfrs_rp.drop(i, inplace = True)
    else:
        dfrs_rh.drop(i, inplace = True)
dfpt_rh = dfpt.copy()
dfpt_rp = dfpt.copy()
for i in dfpt.index:
    if dfpt['RS3_SHARE'][i] == 0:
        dfpt_rp.drop(i, inplace = True)
    else:
        dfpt_rh.drop(i, inplace = True)
fig, axs = plt.subplots(2, 2, figsize=(13, 13))
t = -1
for i in ['CS_1', 'CS_2']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs_rh[i], color="skyblue", label="RS", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt_rh[i], color="red", label="PT", ax=axs[t//2, t%2], 
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
for i in ['CS_1', 'CS_2']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs_rp[i], color="skyblue", label="RS", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(dfpt_rp[i], color="red", label="PT", ax=axs[t//2, t%2], 
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
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - Result 1', 
             x=0.41, y=0.95, fontsize=28)
plt.subplot(2,2,1)
plt.title('SP Ride-hailing', loc='left', fontsize=24)
plt.subplot(2,2,3)
plt.title('SP Ride-pooling', loc='left', fontsize=24)
plt.show()
del t,i,fig,axs,dfrs_rh,dfrs_rp,dfpt_rh,dfpt_rp

# 频率分布直方图 1-5
dfrs_rh = dfrs.copy()
dfrs_rp = dfrs.copy()
for i in dfrs.index:
    if dfrs['RS3_SHARE'][i] == 0:
        dfrs_rp.drop(i, inplace = True)
    else:
        dfrs_rh.drop(i, inplace = True)
dfpt_rh = dfpt.copy()
dfpt_rp = dfpt.copy()
for i in dfpt.index:
    if dfpt['RS3_SHARE'][i] == 0:
        dfpt_rp.drop(i, inplace = True)
    else:
        dfpt_rh.drop(i, inplace = True)
fig, axs = plt.subplots(2, 2, figsize=(13, 13))
t = -1
for i in ['CS_1', 'CS_2']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfrs_rh[i], color="limegreen", label="Ride-hailing", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='forestgreen')
    ax=sns.histplot(dfrs_rp[i], color="sandybrown", label="Ride-pooling", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='chocolate')
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
for i in ['CS_1', 'CS_2']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(dfpt_rh[i], color="limegreen", label="Ride-hailing", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='forestgreen')
    ax=sns.histplot(dfpt_rp[i], color="sandybrown", label="Ride-pooling", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='chocolate')
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
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - Result 2', 
             x=0.42, y=0.95, fontsize=28)
plt.subplot(2,2,1)
plt.title('SP RS', loc='left', fontsize=24)
plt.subplot(2,2,3)
plt.title('SP PT', loc='left', fontsize=24)
plt.show()
del t,i,fig,axs,dfrs_rh,dfrs_rp,dfpt_rh,dfpt_rp


# 频率分布直方图2
biobase.scatterplot(df, 'PT1_TRANS', 'CS_1')


# Scatterplot Matrix
# Create the plot
columns = list(df.columns)
columns.remove('NUM')
#columns.remove('RS_AV')
#columns.remove('PT_AV')
#columns.remove('MODE_2023')
g = sns.PairGrid(data=df, vars=columns, hue=None)
g.map_upper(biobase.reg_coef)
g = g.map_lower(sns.regplot, scatter_kws={"edgecolor": "white"})
g = g.map_diag(sns.histplot, kde=True)
plt.suptitle('Scatterplot Matrix of SP Variables', x=0.12, y=1, fontsize=42)
plt.show()
del columns,g


#%% modeling for rs
df,dfrs,dfpt = biobase.biodata('all', 'sp')
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db
import math

for i in dfrs.index:
    if dfrs['CS_1'][i] == 4:
        dfrs.loc[i,'CS_1'] = 3
    if dfrs['CS_2'][i] == 4:
        dfrs.loc[i,'CS_2'] = 3

for i in dfrs.columns:
    globals()[i] = Variable(i) 
del i

database = db.Database('RS_SP', dfrs)
# Parameters to be estimated
ASC_PT = Beta('ASC_PT', 0, None, None, 0)
ASC_CS = Beta('ASC_CS', 0, None, None, 0)
ASC_RS = Beta('ASC_RS', 0, None, None, 1)

B_WaitT_PT = Beta('B_WaitT_PT', 0, None, None, 0)
B_WalkT_PT = Beta('B_WalkT_PT', 0, None, None, 0)
B_TT_PT = Beta('B_TT_PT', 0, None, None, 0)
B_Cost_PT = Beta('B_Cost_PT', 0, None, None, 0)

B_WalkT_CS = Beta('B_WalkT_CS', 0, None, None, 0)
B_TT_CS = Beta('B_TT_CS', 0, None, None, 0)
B_Cost_CS = Beta('B_Cost_CS', 0, None, None, 0)

B_WaitT_RS = Beta('B_WaitT_RS', 0, None, None, 0)
B_TT_RS = Beta('B_TT_RS', 0, None, None, 0)
B_Cost_RS = Beta('B_Cost_RS', 0, None, None, 0)

B_Trans = Beta('B_Trans', 0, None, None, 0)
B_Crowd = Beta('B_Crowd', 0, None, None, 0)
B_Disin = Beta('B_Disin', 0, None, None, 0)
B_Share = Beta('B_Share', 0, None, None, 0)

B_PTD = Beta('B_PTD', 0, None, None, 0)
B_CSD = Beta('B_CSD', 0, None, None, 0)
B_RSD = Beta('B_RSD', 0, None, None, 1)

# Scaleing variables
COMMUTING_DAYS = COMMUTING_DAYS / 10
PT1_COST = PT1_COST / 1000
CS2_COST = CS2_COST / 1000
RS3_COST = RS3_COST / 1000
PT1_TRANS = PT1_TRANS / 100
PT1_CROWD = PT1_CROWD / 10
CS2_DISIN = CS2_DISIN / 100
RS3_SHARE = RS3_SHARE / 10

PT1_WAIT = PT1_WAIT / 100
PT1_WALK = PT1_WALK / 100
PT1_TT = PT1_TT / 100
CS2_WALK = CS2_WALK / 100
CS2_TT = CS2_TT / 100
RS3_WAIT = RS3_WAIT / 100
RS3_TT = RS3_TT / 100

'''
dfrs.loc[dfrs['CS_1'] == 1]['PT1_TRANS']
biobase.scatterplot(dfrs, 'PT1_TRANS', 'CS_1')

fig, axs = plt.subplots(1, 1)
ax=sns.histplot(dfrs.loc[dfrs['CS_1'] == 1]['PT1_TRANS'], color="skyblue", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='dodgerblue')
ax=sns.histplot(dfrs.loc[dfrs['CS_1'] == 2]['PT1_TRANS'], color="red", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='orangered')
ax=sns.histplot(dfrs.loc[dfrs['CS_1'] == 3]['PT1_TRANS'], color="limegreen", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='forestgreen')
'''

sns.displot(dfrs,x='PT1_TRANS', label="RS", kde=True, 
            hue="CHOICE",stat='probability')



# Definition of the utility functions
#pt
PT = (ASC_PT + B_WaitT_PT*PT1_WAIT + B_WalkT_PT*PT1_WALK + B_TT_PT*PT1_TT + 
      B_Cost_PT*PT1_COST + B_Trans*PT1_TRANS + B_Crowd*PT1_CROWD + 
      B_PTD*COMMUTING_DAYS
      )

#cs
CS = (ASC_CS + B_WalkT_CS*CS2_WALK + B_TT_CS*CS2_TT + 
      B_Cost_CS*CS2_COST + B_Disin*CS2_DISIN + 
      B_CSD*COMMUTING_DAYS
      )

#rs
RS = (ASC_RS + B_WaitT_RS*RS3_WAIT + B_TT_RS*RS3_TT + 
      B_Cost_RS*RS3_COST + B_Share*RS3_SHARE + 
      B_RSD*COMMUTING_DAYS
      )

V1 = PT - CS
V2 = PT - RS
V3 = CS - PT
V4 = CS - RS
V5 = RS - PT
V6 = RS - CS

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3, 4: V4, 5: V5, 6: V6}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, None, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'default_sp_rs' #change the model name here

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
biobase.writeExcel(results, onlyRobust=True, 
    path='CF(sp)/'
)



#%% modeling for pt
df,dfrs,dfpt = biobase.biodata('all', 'sp')
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db

for i in dfpt.index:
    if dfpt['CS_1'][i] == 4:
        dfpt.loc[i,'CS_1'] = 3
    if dfpt['CS_2'][i] == 4:
        dfpt.loc[i,'CS_2'] = 3

for i in dfpt.columns:
    globals()[i] = Variable(i) 
del i

database = db.Database('PT_SP', dfpt)
# Parameters to be estimated
ASC_PT = Beta('ASC_PT', 0, None, None, 0)
ASC_CS = Beta('ASC_CS', 0, None, None, 0)
ASC_RS = Beta('ASC_RS', 0, None, None, 1)

B_WaitT = Beta('B_WaitT', 0, None, None, 0)
B_WalkT = Beta('B_WalkT', 0, None, None, 0)
B_TT = Beta('B_TT', 0, None, None, 0)
B_Cost = Beta('B_Cost', 0, None, None, 0)

B_Trans = Beta('B_Trans', 0, None, None, 0)
B_Crowd = Beta('B_Crowd', 0, None, None, 0)
B_Disin = Beta('B_Disin', 0, None, None, 0)
B_Share = Beta('B_Share', 0, None, None, 0)

B_PTD = Beta('B_PTD', 0, None, None, 0)
B_CSD = Beta('B_CSD', 0, None, None, 0)
B_RSD = Beta('B_RSD', 0, None, None, 1)

# Scaleing variables
COMMUTING_DAYS = COMMUTING_DAYS / 100
PT1_COST = PT1_COST / 10000
CS2_COST = CS2_COST / 10000
RS3_COST = RS3_COST / 10000
PT1_TRANS = PT1_TRANS / 10
PT1_CROWD = PT1_CROWD / 10
CS2_DISIN = CS2_DISIN / 100
RS3_SHARE = RS3_SHARE / 10

PT1_WAIT = PT1_WAIT / 100
PT1_WALK = PT1_WALK / 100
PT1_TT = PT1_TT / 100
CS2_WALK = CS2_WALK / 100
CS2_TT = CS2_TT / 100
RS3_WAIT = RS3_WAIT / 100
RS3_TT = RS3_TT / 100


# Definition of the utility functions
#pt
PT = (ASC_PT + B_WaitT*PT1_WAIT + B_WalkT*PT1_WALK + B_TT*PT1_TT + 
      B_Cost*PT1_COST + B_Trans*PT1_TRANS + B_Crowd*PT1_CROWD + 
      B_PTD*COMMUTING_DAYS
      )

#cs
CS = (ASC_CS + B_WalkT*CS2_WALK + B_TT*CS2_TT + 
      B_Cost*CS2_COST + B_Disin*CS2_DISIN + 
      B_CSD*COMMUTING_DAYS
      )

#rs
RS = (ASC_RS + B_WaitT*RS3_WAIT + B_TT*RS3_TT + 
      B_Cost*RS3_COST + B_Share*RS3_SHARE + 
      B_RSD*COMMUTING_DAYS
      )

V1 = PT - CS
V2 = PT - RS
V3 = CS - PT
V4 = CS - RS
V5 = RS - PT
V6 = RS - CS

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3, 4: V4, 5: V5, 6: V6}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, None, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'default_sp_pt' #change the model name here

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
    path='CF(sp)/'
)


