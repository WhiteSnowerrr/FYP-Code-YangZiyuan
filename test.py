#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:53:51 2023

@author: yangziyuan
"""

import biobase
#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all
df,dfrs,dfpt,travel = biobase.biodata('all', 'subj',traveldata=True)
df = biobase.biodata(row='pt', col='all')
database = biobase.pp(df)

biobase.pp(df)


from biobase import *
database = db.Database('rspt', df)

for i in df.columns:
    del globals() [i]


#%% default modeling

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)
CAR_AV_SP = CAR_AV * (SP != 0)
TRAIN_AV_SP = TRAIN_AV * (SP != 0)
TRAIN_TT_SCALED = TRAIN_TT / 100
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100
SM_COST_SCALED = SM_COST / 100
CAR_TT_SCALED = CAR_TT / 100
CAR_CO_SCALED = CAR_CO / 100

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'default' #change the model name here

'''# Calculate the null log likelihood for reporting.
print('Null log likelihood:',the_biogeme.calculateNullLoglikelihood(av))'''

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
newwriteHtml(results,
    path='/Users/yangziyuan/Documents/学/大四上/fyp/results/'
)


#%%
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
#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all
df,dfrs,dfpt = biobase.biodata('all', 'subj')


#%% summary
# summary table
print(df.head())
describe = df.describe()
print(describe)
describe.to_excel(
    '/Users/yangziyuan/Documents/学/大四上/fyp/FYP CODE/BE(subj)/describe.xlsx')


# scattermatrix
sns.pairplot(df, kind="scatter", hue='MODE_2023', 
             markers=["o", "s"], palette="Set2")
plt.show()


# 交叉频数分析1
ct = pd.crosstab(df['ATT_PT_2'], df['MODE_2023'], normalize=True)
sns.heatmap(ct, cmap='YlOrRd', annot=True, mask=ct<0.1) #图1
from statsmodels.graphics.mosaicplot import mosaic
props = lambda key: {"color": "0.45"} if '4' in key else {"color": "#C6E2FF"}
mosaic(ct.stack(), properties=props) #图2


# 交叉频数分析2 频率分布直方图
sns.distplot(dfrs['ATT_PT_2'], color="skyblue", label="rs", kde=True)
sns.distplot(dfpt['ATT_PT_2'], color="red", label="pt", kde=True)
plt.legend() 
plt.show()

#
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['CSRS_1_1', 'CSRS_1_2', 'ATT_PT_1', 'ATT_PT_2', 'ATT_PT_3', 'ATT_PT_4']:
    t = t+1
    sns.distplot(dfrs[i], color="skyblue", label="rs", kde=True, ax=axs[t//3, t%3])
    sns.distplot(dfpt[i], color="red", label="pt", kde=True, ax=axs[t//3, t%3])
plt.legend() 
plt.show()
del t,i,fig,axs

#
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
t = -1
for i in ['ATT_SHIFT_RS_1', 'ATT_SHIFT_RS_2', 'ATT_SHIFT_RS_3']:
    t = t+1
    sns.distplot(dfrs[i], color="skyblue", label="rs", kde=True, ax=axs[t%3])
plt.legend() 
plt.show()
del t,i,fig,axs


'''# 分面展示多个直方图
# 设置画板
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# 分别绘制多个直方图
sns.distplot(df["sepal_length"], kde=True, color="skyblue", ax=axs[0, 0])
sns.distplot(df["sepal_width"], kde=True, color="olive", ax=axs[0, 1])
sns.distplot(df["petal_length"], kde=True, color="gold", ax=axs[1, 0])
sns.distplot(df["petal_width"], kde=True, color="teal", ax=axs[1, 1])

plt.show()'''


# 散点图
import random
dftem = df.copy()
for i in dftem.index:
    for j in df.columns:
        t = 10**len(str(dftem[j][i]))*0.0001
        dftem[j][i] = dftem[j][i] + random.uniform(-t, t)
del i,j,t

plt.plot( 'ATT_PT_2', 'MODE_2023', 
         data=dftem, linestyle='', marker='o', 
         markersize=3, alpha=0.05, color="red")
plt.xlabel('Value of X') # 设置x轴标签
plt.ylabel('Value of Y') # 设置y轴标签
plt.title('Overplotting looks like that:', loc='left') # 设置标题


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
plt.show()
del columns,g



#%%
from biogeme.expressions import Variable
for i in df.columns:
    globals()[i] = Variable(i) 
del i













#%% modeling for rs
import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df,dfrs,dfpt = biobase.biodata('all', 'sp')
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db

for i in df.columns:
    globals()[i] = Variable(i) 
del i

database = db.Database('RS_SP', dfrs)
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

# Definition of new variables
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
biobase.newwriteHtml(results,
    path='test/'
)


#%% modeling for rs
df,dfrs,dfpt = biobase.biodata('all', 'sp')
from biogeme.expressions import Variable, Beta
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db

for i in dfrs.index:
    if dfrs['CS_1'][i] == 4:
        dfrs.loc[i,'CS_1'] = 3
    if dfrs['CS_2'][i] == 4:
        dfrs.loc[i,'CS_2'] = 3

for i in df.columns:
    globals()[i] = Variable(i) 
del i

database = db.Database('RS_SP', dfrs)
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

# Definition of new variables
COMMUTING_DAYS = COMMUTING_DAYS / 100
PT1_COST = PT1_COST / 1000
CS2_COST = CS2_COST / 1000
RS3_COST = RS3_COST / 1000
PT1_TRANS = PT1_TRANS / 10
PT1_CROWD = PT1_CROWD / 10
CS2_DISIN = CS2_DISIN / 10
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
V1 = (ASC_PT + B_WaitT*PT1_WAIT + B_WalkT*PT1_WALK + B_TT*PT1_TT + 
      B_Cost*PT1_COST + B_Trans*PT1_TRANS + B_Crowd*PT1_CROWD + 
      B_PTD*COMMUTING_DAYS
      )

#cs
V2 = (ASC_CS + B_WalkT*CS2_WALK + B_TT*CS2_TT + 
      B_Cost*CS2_COST + B_Disin*CS2_DISIN + 
      B_CSD*COMMUTING_DAYS
      )

#rs
V3 = (ASC_RS + B_WaitT*RS3_WAIT + B_TT*RS3_TT + 
      B_Cost*RS3_COST + B_Share*RS3_SHARE + 
      B_RSD*COMMUTING_DAYS
      )

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, None, CS_1)

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
biobase.newwriteHtml(results,
    path='CF(sp)/'
)


#%%


import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all

df,dfrs,dfpt = biobase.biodata('all', 'subj')


# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['CSRS_1_1', 'CSRS_1_2', 'ATT_PT_1', 'ATT_PT_2', 'ATT_PT_3', 'ATT_PT_4']:
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
plt.suptitle('Distribution Plots of Subjective Variables', x=0.37, fontsize=28)
plt.show()
del t,i,fig,axs

#%%
import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all

df,dfrs,dfpt = biobase.biodata('all', 'subj')
for i in df.index:
    if df['MODE_2023'][i] == 1:
        df.loc[i,'MODE_2023'] = 'RS'
    else:
        df.loc[i,'MODE_2023'] = 'PT'

# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 3, figsize=(15, 9))

t = 0
for i in ['CSRS_1_1', 'CSRS_1_2', 'ATT_PT_1', 'ATT_PT_2', 'ATT_PT_3', 'ATT_PT_4']:
    
    sns.set_context(rc = {'patch.linewidth': 0.0})
    t = t+1
    plt.subplot(2,3,t)
    sns.displot(df,x=i, color="skyblue", label="RS", kde=True, hue="MODE_2023",stat='probability')

    ticks=[]
    for j in df.index:
        if df[i][j] not in ticks:
            ticks.append(df[i][j])
    ticks.sort()
    step = ticks[1] - ticks[0]
    ticks.append(max(ticks)+step)
    ticks.append(min(ticks)-step)

    del ticks,step,j
plt.legend()
plt.suptitle('Distplots of Subjective Variables', ha='right', fontsize=28)
plt.show()
del t,i,fig,axs

#%%
import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from seaborn import palettes

#row-choose row data:rs/pt/all
#col-choose col data:obj/subj/sp/all

df,dfrs,dfpt = biobase.biodata('all', 'subj')
df,dfrs,dfpt = biobase.biodata('all', 'subj')
for i in df.index:
    if df['MODE_2023'][i] == 1:
        df.loc[i,'MODE_2023'] = 'RS'
    else:
        df.loc[i,'MODE_2023'] = 'PT'

col = palettes._ColorPalette(['skyblue', 'red'])

# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['CSRS_1_1', 'CSRS_1_2', 'ATT_PT_1', 'ATT_PT_2', 'ATT_PT_3', 'ATT_PT_4']:
    if t != -1:
        ax.get_legend().remove()
    sns.set_palette(col)
    t = t+1
    ax=sns.histplot(df,x=i, hue="MODE_2023", ax=axs[t//3, t%3], 
                    stat='probability', discrete=True, shrink=.5, kde=True)
    ticks=[]
    for j in df.index:
        if df[i][j] not in ticks:
            ticks.append(df[i][j])
    ticks.sort()
    step = ticks[1] - ticks[0]
    #ticks.append(max(ticks)+step)
    #ticks.append(min(ticks)-step)
    ax.set_xticks(ticks)
    #ax.set_ylim(0,1)

    del ticks,step,j
plt.suptitle('Distplots of Subjective Variables', ha='right', fontsize=28)
plt.show()
del t,i,fig,axs,ax

#%%
import pandas as pd
import numpy as np
import scipy
temp = pd.read_pickle('data/555.pickle')['self'].data.H
temp2 = np.nan_to_num(temp)
temp3 = scipy.linalg.pinv(temp2)
temp4 = scipy.linalg.inv(temp2)
temp5 = scipy.linalg.pinv(temp2, rtol = np.spacing(0.01))


out1 = np.dot(temp2,temp3)
out2 = np.dot(temp2,temp4)


'''
temp00 = np.array([[1,0],[0,0]])
np.linalg.inv(temp00)
'''

#%%

import pandas as pd
data = pd.io.stata.read_stata('data/40273_2017_506_MOESM3_ESM.dta')

#%%
import pandas as pd
data_temp = pd.read_pickle('data/rspt6.pickle')

df2 = data_temp['df']
travel2 = data_temp['travel']
del data_temp


for i in [1744]: #here
    t = i - i%8
    for ii in range(t,t+8):
        df2.drop(ii, inplace = True)
        travel2.drop(ii, inplace = True)
del i,t,ii




for i in df2.index:
    if i%8 == 0 and len(set(df2['cs_1'][range(i,i+8)])) == 1:
        for ii in range(i,i+8):
            df2.drop(ii, inplace = True)
            travel2.drop(ii, inplace = True)
del i,ii



for i in df2.index:
    if i%8 == 0 and len(set(df2['cs_2'][range(i,i+8)])) == 1:
        for ii in range(i,i+8):
            df2.drop(ii, inplace = True)
            travel2.drop(ii, inplace = True)
del i,ii


for i in df2.index:
    if i%8 == 0 and len(set(df2['cs_1'][range(i,i+8)])) == 1 and len(set(df2['cs_2'][range(i,i+8)])) == 1:
        for ii in range(i,i+8):
            df2.drop(ii, inplace = True)
            travel2.drop(ii, inplace = True)
del i,ii


#%%
import biobase
import pandas as pd
df0,dfrs0,dfpt0,travel = biobase.biodata('all', 'all',traveldata=True)

df0['PT_WT'] = 0
df0['PT_TT'] = 0


for i in travel.index:
    temp = travel['pt_now'][i]['legs'][0]['steps']
    walkT = 0
    TT = 0
    for j in temp:
        if j['travel_mode'] == 'WALKING':
            walkT += j['duration']['value']/60
        elif j['travel_mode'] == 'TRANSIT':
            TT += j['duration']['value']/60
    df0.loc[i,'PT_WT'] = walkT * df0.loc[i,'PT1_WALK']/df0.loc[i,'PT_WALKING']
    df0.loc[i,'PT_TT'] = TT * df0.loc[i,'PT1_TT']/df0.loc[i,'PT_TIME']
        
del i,j,TT,walkT,dfrs0,dfpt0,temp
    

writer = pd.ExcelWriter('test/describe.xlsx')
df0.to_excel(writer, sheet_name='All')

writer._save()

biobase.resetCol('test/describe.xlsx')


#%%
import biobase
import pandas as pd

df0,dfrs0,dfpt0 = biobase.biodata('all', 'all')


temp = pd.DataFrame()
for ii in [1,2]:
    for jj in [1,2,3,4]:
        a=0
        b=0
        for i in df0.index:
            if i%8 == 0:
                b += 1
                if len(set(df0['CS_'+str(ii)][range(i,i+8)])) <= jj:
                    a += 1

        temp.loc['ALL',str(ii)+'_'+str(jj)] = round(a/b*100,2)
        a=0
        b=0
        for i in dfrs0.index:
            if i%8 == 0:
                b += 1
                if len(set(dfrs0['CS_'+str(ii)][range(i,i+8)])) <= jj:
                    a += 1
                
        temp.loc['RS',str(ii)+'_'+str(jj)] = round(a/b*100,2)
        a=0
        b=0
        for i in dfpt0.index:
            if i%8 == 0:
                b += 1
                if len(set(dfpt0['CS_'+str(ii)][range(i,i+8)])) <= jj:
                    a += 1
                
        temp.loc['PT',str(ii)+'_'+str(jj)] = round(a/b*100,2)


temp2 = pd.DataFrame()
for i in ['ALL','RS','PT']:
    for j in [1,2,3,4]:
        temp2.loc[i,str(j)] = 0

for i in df0.index:
    ii = df0['CS_1'][i]
    temp2.loc['ALL',str(ii)] += 1
for i in dfrs0.index:
    ii = dfrs0['CS_1'][i]
    temp2.loc['RS',str(ii)] += 1
for i in dfpt0.index:
    ii = dfpt0['CS_1'][i]
    temp2.loc['PT',str(ii)] += 1

temp3 = temp2.copy()
temp3['3'] = temp3['3']+temp3['4']
temp3.drop('4', axis=1,inplace = True)
temp3['all'] = temp3['3']+temp3['2']+temp3['1']

temp3['1'] = round(temp3['1']/temp3['all']*100,2)
temp3['2'] = round(temp3['2']/temp3['all']*100,2)
temp3['3'] = round(temp3['3']/temp3['all']*100,2)


temp2 = pd.DataFrame()
for i in ['ALL','RS','PT']:
    for j in [1,2,3,4]:
        temp2.loc[i,str(j)] = 0

for i in df0.index:
    if i%8 == 0 and len(set(df0['CS_1'][range(i,i+8)])) == 1:
        ii = df0['CS_1'][i]
        temp2.loc['ALL',str(ii)] += 1
    
for i in dfrs0.index:
    if i%8 == 0 and len(set(dfrs0['CS_1'][range(i,i+8)])) == 1:
        ii = dfrs0['CS_1'][i]
        temp2.loc['RS',str(ii)] += 1
    
for i in dfpt0.index:
    if i%8 == 0 and len(set(dfpt0['CS_1'][range(i,i+8)])) == 1:
        ii = dfpt0['CS_1'][i]
        temp2.loc['PT',str(ii)] += 1
    

temp3 = temp2.copy()
temp3['3'] = temp3['3']+temp3['4']
temp3.drop('4', axis=1,inplace = True)
temp3['all'] = temp3['3']+temp3['2']+temp3['1']

temp3['1'] = round(temp3['1']/temp3['all']*100,2)
temp3['2'] = round(temp3['2']/temp3['all']*100,2)
temp3['3'] = round(temp3['3']/temp3['all']*100,2)

temp4 = temp3.copy()






temp2 = pd.DataFrame()
for i in ['ALL','RS','PT']:
    for j in [1,2,3,4]:
        temp2.loc[i,str(j)] = 0

for i in df0.index:
    if i%8 == 0 and len(set(df0['CS_2'][range(i,i+8)])) == 1:
        ii = df0['CS_2'][i]
        temp2.loc['ALL',str(ii)] += 1
    
for i in dfrs0.index:
    if i%8 == 0 and len(set(dfrs0['CS_2'][range(i,i+8)])) == 1:
        ii = dfrs0['CS_2'][i]
        temp2.loc['RS',str(ii)] += 1
    
for i in dfpt0.index:
    if i%8 == 0 and len(set(dfpt0['CS_2'][range(i,i+8)])) == 1:
        ii = dfpt0['CS_2'][i]
        temp2.loc['PT',str(ii)] += 1
    

temp3 = temp2.copy()
temp3['3'] = temp3['3']+temp3['4']
temp3.drop('4', axis=1,inplace = True)
temp3['all'] = temp3['3']+temp3['2']+temp3['1']

temp3['1'] = round(temp3['1']/temp3['all']*100,2)
temp3['2'] = round(temp3['2']/temp3['all']*100,2)
temp3['3'] = round(temp3['3']/temp3['all']*100,2)

#%%
import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df0,dfrs0,dfpt0 = biobase.biodata('all', 'all')

a = dfrs0[dfrs0['CS_2'].isin([1])]
b = dfrs0[dfrs0['CS_2'].isin([3,4])]

# 交叉频数分析2 频率分布直方图 1-1
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
t = -1
for i in ['PT1_WAIT', 'PT1_WALK', 'PT1_TT', 
          'PT1_COST', 'PT1_TRANS', 'PT1_CROWD']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(a[i], color="skyblue", label="a", ax=axs[t//3, t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(b[i], color="red", label="b", ax=axs[t//3, t%3], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')

    del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - PT', 
             x=0.34, y=0.94, fontsize=28)
plt.show()
del t,i,fig,axs



fig, axs = plt.subplots(2, 2, figsize=(9, 9))
t = -1
for i in ['RS3_WAIT', 'RS3_TT', 
          'RS3_COST', 'RS3_SHARE']:
    sns.set_context(rc = {'patch.linewidth': 1.42})
    t = t+1
    ax=sns.histplot(a[i], color="skyblue", label="a", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='dodgerblue')
    ax=sns.histplot(b[i], color="red", label="b", ax=axs[t//2, t%2], 
                    stat='density', discrete=True, shrink=.5, kde=True, 
                    kde_kws={'cut':1.8}, edgecolor='orangered')
    del ax
sns.set_context(rc = {'patch.linewidth': 1.0})
plt.legend()
plt.suptitle('Distribution Plots of SP Variables - RS', 
             x=0.34, y=0.92, fontsize=20)
plt.show()
del t,i,fig,axs


ax=sns.histplot(a['CS_1'], color="skyblue", label="RS",  
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='dodgerblue')
ax=sns.histplot(b['CS_1'], color="red", label="PT", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='orangered')




ax=sns.histplot(a['COMMUTING_DAYS'], color="skyblue", label="RS",  
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='dodgerblue')
ax=sns.histplot(b['COMMUTING_DAYS'], color="red", label="PT", 
                stat='density', discrete=True, shrink=.5, kde=True, 
                kde_kws={'cut':1.8}, edgecolor='orangered')


#%%
import biobase
import biowrite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df,dfrs,dfpt = biobase.biodata('all', 'all')
# summary table
print(df.head())
describe = df.describe()
print(describe)
describe_rs = dfrs.describe()
describe_pt = dfpt.describe()
writer = pd.ExcelWriter('test/describe.xlsx')
describe.to_excel(writer, sheet_name='All')
describe_rs.to_excel(writer, sheet_name='RS')
describe_pt.to_excel(writer, sheet_name='PT')
writer._save()
del describe,describe_rs,describe_pt,writer
biowrite.resetCol('test/describe.xlsx')


#%%
import biobase
import biowrite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df,dfrs,dfpt = biobase.biodata('all', 'all')

s1=0
s2=0
s3=0
s4=0
for i in dfrs.index:
    t = 0
    if dfrs.loc[i,'ATT_SHIFT_RS_1']==3:
        t = 1
        s1 += 1
    if dfrs.loc[i,'ATT_SHIFT_RS_2']==3:
        t = 1
        s2 += 1
    if dfrs.loc[i,'ATT_SHIFT_RS_3']==3:
        t = 1
        s3 += 1
    if t ==1:
        s4 += 1


plt.pie(x=[s1,s2,s3,s4],
        labels=['1','2','3','other'],
        #colors=["#d5695d", "#5d8ca8", "#65a479"], 
        autopct='%.2f%%',#格式化输出百分比
       )
plt.legend(['1','2','3','other'],#添加图例
          title="f3",
          loc="center left",
          fontsize=15,
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()

#%%
import biobase
import biowrite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df,dfrs,dfpt = biobase.biodata('all', 'all')

rs1=0
rs2=0
rs3=0

for i in dfrs.index:
    if dfrs.loc[i,'CS_1']==1:
        rs1 += 1
    if dfrs.loc[i,'CS_1']==2:
        rs2 += 1
    if dfrs.loc[i,'CS_1']==3:
        rs3 += 1

pt1=0
pt2=0
pt3=0

for i in dfpt.index:
    if dfpt.loc[i,'CS_1']==1:
        pt1 += 1
    if dfpt.loc[i,'CS_1']==2:
        pt2 += 1
    if dfpt.loc[i,'CS_1']==3:
        pt3 += 1

rst=rs1+rs2+rs3
rs1 = rs1/rst
rs2 = rs2/rst
rs3 = rs3/rst

ptt=pt1+pt2+pt3
pt1=pt1/ptt
pt2=pt2/ptt
pt3=pt3/ptt

#plt.rcParams['font.family'] = "Times New Roman"
fig,ax = plt.subplots()
 
label = ['rs','pt']
pt_number = [rs1,pt1]
cs_number = [rs2,pt2]
rs_number = [rs3,pt3]
a = [rs1+rs2,pt1+pt2]
 
width = .4
 
ax.bar(label, pt_number, width, label='PT',color='white',hatch="//",ec='k',lw=.6)
ax.bar(label, cs_number, width,  bottom=pt_number, label='CS',color='gray',ec='k',lw=.6)
ax.bar(label, rs_number, width,  bottom=a, label='RS',color='white',hatch="...",ec='k',lw=.6)

ax.set_ylim(0,1.2)

ax.tick_params(direction='out',labelsize=12,length=5.5,width=1,top=False,right=False)
ax.legend(fontsize=11,frameon=False,loc='upper center',ncol=4)
ax.set_ylabel('',fontsize=13)
ax.set_xlabel('people group',fontsize=13)
text_font = {'size':'17','weight':'bold','color':'black'}


#plt.savefig(r'F:\DataCharm\SCI paper plots\sci_bar_04.png',width=5,height=3,


#%%
import biobase
import biowrite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df,dfrs,dfpt = biobase.biodata('all', 'all')

rs1=0
rs2=0
rs3=0

for i in dfrs.index:
    if dfrs.loc[i,'CS_2']==1:
        rs1 += 1
    if dfrs.loc[i,'CS_2']==2:
        rs2 += 1
    if dfrs.loc[i,'CS_2']==3:
        rs3 += 1

pt1=0
pt2=0
pt3=0

for i in dfpt.index:
    if dfpt.loc[i,'CS_2']==1:
        pt1 += 1
    if dfpt.loc[i,'CS_2']==2:
        pt2 += 1
    if dfpt.loc[i,'CS_2']==3:
        pt3 += 1

rst=rs1+rs2+rs3
rs1 = rs1/rst
rs2 = rs2/rst
rs3 = rs3/rst

ptt=pt1+pt2+pt3
pt1=pt1/ptt
pt2=pt2/ptt
pt3=pt3/ptt

#plt.rcParams['font.family'] = "Times New Roman"
fig,ax = plt.subplots()
 
label = ['rs','pt']
pt_number = [rs1,pt1]
cs_number = [rs2,pt2]
rs_number = [rs3,pt3]
a = [rs1+rs2,pt1+pt2]
 
width = .4
 
ax.bar(label, pt_number, width, label='PT',color='white',hatch="//",ec='k',lw=.6)
ax.bar(label, cs_number, width,  bottom=pt_number, label='CS',color='gray',ec='k',lw=.6)
ax.bar(label, rs_number, width,  bottom=a, label='RS',color='white',hatch="...",ec='k',lw=.6)

ax.set_ylim(0,1.2)

ax.tick_params(direction='out',labelsize=12,length=5.5,width=1,top=False,right=False)
ax.legend(fontsize=11,frameon=False,loc='upper center',ncol=4)
ax.set_ylabel('',fontsize=13)
ax.set_xlabel('people group',fontsize=13)
text_font = {'size':'17','weight':'bold','color':'black'}


#plt.savefig(r'F:\DataCharm\SCI paper plots\sci_bar_04.png',width=5,height=3,

#%%

import biobase
import biowrite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df,dfrs,dfpt = biobase.biodata('all', 'all')


meanIncome = np.mean(df['INCOME_2'])


dfpt['a']=0
dfpt.loc[dfpt['INCOME_2']>meanIncome,'a']=1

sum(dfpt['INCOME_2']>meanIncome)/len(dfpt.index)

sum(dfrs['INCOME_2']>meanIncome)/len(dfrs.index)









