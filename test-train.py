#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:45:33 2023

@author: yangziyuan
"""

import biobase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from biogeme import models
import biogeme.biogeme as bio
import biogeme.database as db
import math
import biomodel
from biomodel import V
from tqdm import tqdm
df,dfrs,dfpt = biobase.biodata('all', 'all')


## modeling for rs
dftemp = df.copy()


df_train, df_test = biobase.trainTestSplit(dftemp, test_size=0.2, random_state=20011207)

the_biogeme,results = biomodel.modelResult(df_train)

pandasResults0 = results.getEstimatedParameters()
print(pandasResults0)



#%% predict
prob_PT = models.logit(V, None, 1)
prob_CS = models.logit(V, None, 2)
prob_RS = models.logit(V, None, 3)

simulate ={'Prob. PT':  prob_PT ,
           'Prob. CS':  prob_CS ,
           'Prob. RS':  prob_RS ,}

database_test = db.Database('SP_test', df_test)
the_biogeme_test = bio.BIOGEME(database_test, simulate)
the_biogeme_test.modelName = "default_sp_test"

betaValues = results.getBetaValues()

simulatedValues = the_biogeme_test.simulate(betaValues)
print(simulatedValues.head())

prob_max = simulatedValues.idxmax(axis=1)
prob_max = prob_max.replace({'Prob. PT': 1, 'Prob. CS': 2, 'Prob. RS': 3})

data = {'y_Actual':    df_test['CS_1'],
        'y_Predicted': prob_max
        }

AP = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(AP['y_Actual'], AP['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])


accuracy = np.diagonal(confusion_matrix.to_numpy()).sum()/confusion_matrix.to_numpy().sum()
print('Global accuracy of the model:', accuracy)


#%% market
from biomodel import PT1_COST
database = db.Database('Data', df_test)

prob_PT = models.logit(V, None, 1)
prob_CS = models.logit(V, None, 2)
prob_RS = models.logit(V, None, 3)

simulate ={'Prob. PT':  prob_PT ,
           'Prob. CS':  prob_CS ,
           'Prob. RS':  prob_RS ,
           'Revenue public transportation': prob_PT * PT1_COST}

the_biogeme_market = bio.BIOGEME(database, simulate)
the_biogeme_market.modelName = "default_sp_market"
betaValues = results.getBetaValues()

betas = the_biogeme_market.free_beta_names()

simulatedValues = the_biogeme_market.simulate(betaValues)
b = results.getBetasForSensitivityAnalysis(betas , size=100, useBootstrap=False)

left, right = the_biogeme_market.confidenceIntervals(b, .9)

marketShare_PT = simulatedValues['Prob. PT'].mean()
marketShare_PT_left = left['Prob. PT'].mean()
marketShare_PT_right = right['Prob. PT'].mean()

marketShare_CS = simulatedValues['Prob. CS'].mean()
marketShare_CS_left = left['Prob. CS'].mean()
marketShare_CS_right = right['Prob. CS'].mean()

marketShare_RS = simulatedValues['Prob. RS'].mean()
marketShare_RS_left = left['Prob. RS'].mean()
marketShare_RS_right = right['Prob. RS'].mean()

print(f"Market Share for PT :  {100*marketShare_PT:.1f}%  [{100*marketShare_PT_left:.1f} % , {100*marketShare_PT_right:.1f} %]")
print(f"Market Share for CS :  {100*marketShare_CS:.1f}%  [{100*marketShare_CS_left:.1f} % , {100*marketShare_CS_right:.1f} %]")
print(f"Market Share for RS :  {100*marketShare_RS:.1f}%  [{100*marketShare_RS_left:.1f} % , {100*marketShare_RS_right:.1f} %]")

revenues_pt = ( simulatedValues['Revenue public transportation']).sum()
revenues_pt_left = (left['Revenue public transportation']).sum()
revenues_pt_right = ( right ['Revenue public transportation']).sum()

print( f"Revenues for PT : {revenues_pt:.3f} [{revenues_pt_left:.3f}, {revenues_pt_right:.3f}]")




def scenario(delta_cost):
    from biomodel import RS3_COST_S,PT,CS,RS,B_Cost,INCOME_2_S
    RS3_COST_Scenario = RS3_COST_S + delta_cost / 1000
    
    # Definition of the utility functions
    #pt
    PT0 = PT

    #cs
    CS0 = CS

    #rs
    RS0 = RS - B_Cost*RS3_COST_S*(1+1/INCOME_2_S) + B_Cost*RS3_COST_Scenario*(1+1/INCOME_2_S)

    V10 = PT0
    V20 = CS0
    V30 = RS0

    # Associate utility functions with the numbering of alternatives
    V0 = {1: V10, 2: V20, 3: V30}
    prob_RS = models.logit(V0, None, 3)
    simulate ={'Prob. RS':  prob_RS ,
               'Revenue RS': prob_RS * RS3_COST_Scenario}
    the_biogeme_revenue_scenarios = bio.BIOGEME(database, simulate)
    betas = the_biogeme_revenue_scenarios.free_beta_names()
    betaValues = results.getBetaValues(betas)
    simulatedValues = the_biogeme_revenue_scenarios.simulate(betaValues)
    marketShare_RS = simulatedValues['Prob. RS'].mean()
    return marketShare_RS



scales = range(-150,150,10)
revenues = [scenario(s) for s in tqdm(scales)]

max_index = np.argmax(revenues)
max_x = scales[max_index]
max_y = [x*100 for x in revenues][max_index]

plt.plot(scales,[x*100 for x in revenues], color='skyblue', linewidth=2)
plt.scatter(max_x, max_y, color='red', s=12) 
plt.annotate(f'Highest Point: ({max_x:}, {max_y:.2f}%)', xy=(max_x, max_y), 
             xytext=(max_x+5, max_y+2),
             arrowprops=dict(arrowstyle='->'))
plt.xlim((min(scales)-(max(scales)-min(scales))*0.05, 
          max(scales)+(max(scales)-min(scales))*0.05))
plt.ylim((0, 
          max([x*100 for x in revenues])+(max([x*100 for x in revenues])-min([x*100 for x in revenues]))*0.35))

plt.vlines(x=0, ymin=0, ymax=100*revenues[int(np.argwhere(np.array(scales)==0))], 
           linestyle='--',color='black',linewidth=0.8)
plt.hlines(y=100*revenues[int(np.argwhere(np.array(scales)==0))], 
           xmin=min(scales)-(max(scales)-min(scales))*0.05, xmax=0, 
           linestyle='--',color='black',linewidth=0.8)

plt.scatter(0, 100*revenues[int(np.argwhere(np.array(scales)==0))], color='limegreen', s=12) 
plt.annotate(f'Default Point: ({0:}, {100*revenues[int(np.argwhere(np.array(scales)==0))]:.2f}%)', 
             xy=(0, 100*revenues[int(np.argwhere(np.array(scales)==0))]), 
             xytext=(0+5, 100*revenues[int(np.argwhere(np.array(scales)==0))]+2),
             arrowprops=dict(arrowstyle='->'))

plt.xlabel("Modification of the price of RS (delta)")
plt.ylabel("Market Share for RS")

plt.show()





#%% revenue-scenarios
database = db.Database('Data', df_test)
def scenario(scale):
    from biomodel import RS3_COST_S,PT,CS,RS,B_Cost,INCOME_2_S
    RS3_COST_Scenario = RS3_COST_S * scale
    
    # Definition of the utility functions
    #pt
    PT0 = PT

    #cs
    CS0 = CS

    #rs
    RS0 = RS - B_Cost*RS3_COST_S*(1+1/INCOME_2_S) + B_Cost*RS3_COST_Scenario*(1+1/INCOME_2_S)

    V10 = PT0
    V20 = CS0
    V30 = RS0

    # Associate utility functions with the numbering of alternatives
    V0 = {1: V10, 2: V20, 3: V30}
    prob_RS = models.logit(V0, None, 3)
    simulate ={'Prob. RS':  prob_RS ,
               'Revenue RS': prob_RS * RS3_COST_Scenario}
    the_biogeme_revenue_scenarios = bio.BIOGEME(database, simulate)
    betas = the_biogeme_revenue_scenarios.free_beta_names()
    betaValues = results.getBetaValues(betas)
    simulatedValues = the_biogeme_revenue_scenarios.simulate(betaValues)
    revenues_rs = (simulatedValues['Revenue RS']).sum()
    return revenues_rs


scales = np.arange(0.0,8,0.01)
revenues = [scenario(s) for s in tqdm(scales)]

max_index = np.argmax(revenues)
max_x = scales[max_index]
max_y = revenues[max_index]

plt.plot(scales,revenues, color='skyblue', linewidth=2)
plt.scatter(max_x, max_y, color='red', s=12) 
plt.annotate(f'Highest Point: ({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y), 
             xytext=(max_x+0.5, max_y+0.5),
             arrowprops=dict(arrowstyle='->'))

plt.vlines(x=1, ymin=0, ymax=8, linestyle='--',color='black',linewidth=0.8)
plt.xlabel("Modification of the price of RS (scale)")
plt.ylabel("Revenues")

plt.show()



#%% elasticities
from biogeme.expressions import Derive
from biomodel import PT1_TT,PT1_COST,CS2_TT,CS2_COST,RS3_TT,RS3_COST
database = db.Database('Data', df_test)
prob_PT = models.logit(V, None, 1)
prob_CS = models.logit(V, None, 2)
prob_RS = models.logit(V, None, 3)


# Direct Elasticities
direct_elas_pt_time  = Derive(prob_PT,'PT1_TT') * PT1_TT / prob_PT
direct_elas_pt_cost  = Derive(prob_PT,'PT1_COST') * PT1_COST / prob_PT
direct_elas_cs_time  = Derive(prob_CS,'CS2_TT') * CS2_TT / prob_CS
direct_elas_cs_cost  = Derive(prob_CS,'CS2_COST') * CS2_COST / prob_CS
direct_elas_rs_time  = Derive(prob_RS,'RS3_TT') * RS3_TT / prob_RS
direct_elas_rs_cost  = Derive(prob_RS,'RS3_COST') * RS3_COST / prob_RS

simulate ={'Prob. PT':  prob_PT ,
           'Prob. CS':  prob_CS ,
           'Prob. RS':  prob_RS ,
           'direct_elas_pt_time':direct_elas_pt_time,
           'direct_elas_pt_cost':direct_elas_pt_cost,
           'direct_elas_cs_time':direct_elas_cs_time,
           'direct_elas_cs_cost':direct_elas_cs_cost,
           'direct_elas_rs_time':direct_elas_rs_time,
           'direct_elas_rs_cost':direct_elas_rs_cost
            }

the_biogeme_elasticities = bio.BIOGEME(database, simulate)
the_biogeme_elasticities.modelName = "default_sp_elasticities_direct"
betaValues = results.getBetaValues()
simulatedValues = the_biogeme_elasticities.simulate(betaValues)

denominator_pt = simulatedValues['Prob. PT'].sum()
denominator_cs = simulatedValues['Prob. CS'].sum()
denominator_rs = simulatedValues['Prob. RS'].sum()

direct_elas_term_pt_time = (simulatedValues['Prob. PT'] * simulatedValues['direct_elas_pt_time'] / denominator_pt).sum()
direct_elas_term_pt_cost = (simulatedValues['Prob. PT'] * simulatedValues['direct_elas_pt_cost'] / denominator_pt).sum()
direct_elas_term_cs_time = (simulatedValues['Prob. CS'] * simulatedValues['direct_elas_cs_time'] / denominator_cs).sum()
direct_elas_term_cs_cost = (simulatedValues['Prob. CS'] * simulatedValues['direct_elas_cs_cost'] / denominator_cs).sum()
direct_elas_term_rs_time = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_rs_time'] / denominator_rs).sum()
direct_elas_term_rs_cost = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_rs_cost'] / denominator_rs).sum()

print(f"Aggregate direct elasticity of pt wrt time: {direct_elas_term_pt_time:.3g}")
print(f"Aggregate direct elasticity of pt wrt cost: {direct_elas_term_pt_cost:.3g}")
print(f"Aggregate direct elasticity of cs wrt time: {direct_elas_term_cs_time:.3g}")
print(f"Aggregate direct elasticity of cs wrt cost: {direct_elas_term_cs_cost:.3g}")
print(f"Aggregate direct elasticity of rs wrt time: {direct_elas_term_rs_time:.3g}")
print(f"Aggregate direct elasticity of rs wrt cost: {direct_elas_term_rs_cost:.3g}")


# Cross Elasticities
cross_elas_pt_time = Derive(prob_PT,'RS3_TT') * RS3_TT / prob_PT
cross_elas_pt_cost = Derive(prob_PT,'RS3_COST') * RS3_COST / prob_PT
cross_elas_rs_time = Derive(prob_RS,'PT1_TT') * PT1_TT / prob_RS
cross_elas_rs_cost = Derive(prob_RS,'PT1_COST') * PT1_COST / prob_RS

simulate ={'Prob. PT':  prob_PT ,
           'Prob. RS':  prob_RS ,
           'cross_elas_pt_time':cross_elas_pt_time,
           'cross_elas_pt_cost':cross_elas_pt_cost,
           'cross_elas_rs_time':cross_elas_rs_time,
           'cross_elas_rs_cost':cross_elas_rs_cost
            }

the_biogeme_elasticities = bio.BIOGEME(database, simulate)
the_biogeme_elasticities.modelName = "default_sp_elasticities_cross"
betaValues = results.getBetaValues()
simulatedValues = the_biogeme_elasticities.simulate(betaValues)

denominator_pt = simulatedValues['Prob. PT'].sum()
denominator_rs = simulatedValues['Prob. RS'].sum()

cross_elas_term_pt_time = (simulatedValues['Prob. PT'] * simulatedValues['cross_elas_pt_time'] / denominator_pt).sum()
cross_elas_term_pt_cost = (simulatedValues['Prob. PT'] * simulatedValues['cross_elas_pt_cost'] / denominator_pt).sum()
cross_elas_term_rs_time = (simulatedValues['Prob. RS'] * simulatedValues['cross_elas_rs_time'] / denominator_rs).sum()
cross_elas_term_rs_cost = (simulatedValues['Prob. RS'] * simulatedValues['cross_elas_rs_cost'] / denominator_rs).sum()

print(f"Aggregate cross elasticity of PT wrt RS time: {cross_elas_term_pt_time:.3g}")
print(f"Aggregate cross elasticity of PT wrt RS cost: {cross_elas_term_pt_cost:.3g}")
print(f"Aggregate cross elasticity of RS wrt PT time: {cross_elas_term_rs_time:.3g}")
print(f"Aggregate cross elasticity of RS wrt PT cost: {cross_elas_term_rs_cost:.3g}")


# Arc Elasticities

def scenario(delta_cost):
    from biomodel import RS3_COST_S,PT,CS,RS,B_Cost,INCOME_2_S,RS3_COST
    RS3_COST_Scenario = (RS3_COST + delta_cost) / 1000
    
    # Definition of the utility functions
    #pt
    PT0 = PT

    #cs
    CS0 = CS

    #rs
    RS0 = RS - B_Cost*RS3_COST_S*(1+1/INCOME_2_S) + B_Cost*RS3_COST_Scenario*(1+1/INCOME_2_S)

    V10 = PT0
    V20 = CS0
    V30 = RS0

    # Associate utility functions with the numbering of alternatives
    V0 = {1: V10, 2: V20, 3: V30}
    prob_RS = models.logit(V, None, 3)
    prob_RS_after = models.logit(V0, None, 3)
    direct_elas_RS_cost = (prob_RS_after - prob_RS) * RS3_COST / (prob_RS * delta_cost)
    simulate ={'Prob. RS':  prob_RS ,
               'Prob. RS after':  prob_RS_after ,
               'direct_elas_RS_cost': direct_elas_RS_cost}
    the_biogeme_elasticities = bio.BIOGEME(database, simulate)
    the_biogeme_elasticities.modelName = "default_sp_elasticities_arc"
    betaValues = results.getBetaValues()
    simulatedValues = the_biogeme_elasticities.simulate(betaValues)
    denominator_rs = simulatedValues['Prob. RS'].sum()
    direct_elas_RS_cost = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_RS_cost'] / denominator_rs).sum()
    
    #print(f"Aggregate direct elasticity of RS wrt cost: {direct_elas_RS_cost:.3g}")
    return direct_elas_RS_cost

print(f"Aggregate direct elasticity of RS wrt distance: {scenario(0.1):.3g}")

'''
scales = np.arange(0.1,5,0.1).round(5)
revenues = [scenario(s) for s in scales]
plt.plot(scales,revenues)
plt.xlabel("Arc Elasticities of the cost of RS")
plt.ylabel("elasticities")
plt.show()
'''



# Elasticities Change
def scenario(scale):
    from biomodel import RS3_COST_S,PT,CS,RS,B_Cost,INCOME_2_S,RS3_COST
    RS3_COST_Se = RS3_COST * scale
    RS3_COST_Scenario = RS3_COST_Se / 1000
    # Definition of the utility functions
    #pt
    PT0 = PT

    #cs
    CS0 = CS

    #rs
    RS0 = RS - B_Cost*RS3_COST_S*(1+1/INCOME_2_S) + B_Cost*RS3_COST_Scenario*(1+1/INCOME_2_S)

    V10 = PT0
    V20 = CS0
    V30 = RS0

    # Associate utility functions with the numbering of alternatives
    V0 = {1: V10, 2: V20, 3: V30}
    prob_RS = models.logit(V0, None, 3)
        
    direct_elas_rs_cost  = Derive(prob_RS,'RS3_COST') * RS3_COST_Se / prob_RS

    simulate ={
               'Prob. RS':  prob_RS ,
               'direct_elas_rs_cost':direct_elas_rs_cost
                }

    the_biogeme_elasticities = bio.BIOGEME(database, simulate)
    the_biogeme_elasticities.modelName = "default_sp_elasticities_direct"
    betaValues = results.getBetaValues()
    simulatedValues = the_biogeme_elasticities.simulate(betaValues)

    denominator_rs = simulatedValues['Prob. RS'].sum()

    direct_elas_term_rs_cost = (simulatedValues['Prob. RS'] * simulatedValues['direct_elas_rs_cost'] / denominator_rs).sum()

    
    
    #print(f"Aggregate direct elasticity of RS wrt cost: {direct_elas_RS_cost:.3g}")
    return direct_elas_term_rs_cost


scales = np.arange(0.1,5,0.1).round(5)
revenues = [scenario(s) for s in tqdm(scales)]
plt.plot(scales,revenues, color='skyblue')
plt.xlabel("Elasticities of the cost of RS")
plt.ylabel("elasticities")
plt.show()







#%% wtp
from biogeme.expressions import Derive
prob_PT = models.logit(V, None, 1)
prob_RS = models.logit(V, None, 3)
WTP_PT_TIME = Derive(prob_PT,'PT1_TT') / Derive(prob_PT,'PT1_COST')
WTP_RS_TIME = Derive(prob_RS,'RS3_TT') / Derive(prob_RS,'RS3_COST')

simulate = {'WTP PT time': WTP_PT_TIME,
            'WTP RS time': WTP_RS_TIME}

the_biogeme_wtp = bio.BIOGEME(database, simulate, removeUnusedVariables=False)
the_biogeme_wtp.modelName = "default_sp_wtp"
betas = the_biogeme_wtp.free_beta_names()
betaValues = results.getBetaValues()
simulatedValues = the_biogeme_wtp.simulate(betaValues)

wtprs = (60 * simulatedValues['WTP RS time']).mean()

b = results.getBetasForSensitivityAnalysis(betas,size=100, useBootstrap=False)
left,right = the_biogeme_wtp.confidenceIntervals(b,0.9)
wtprs_left = (60 * left['WTP RS time']).mean()
wtprs_right = (60 * right['WTP RS time']).mean()

print(f"Average WTP for RS: {wtprs:.3g} CI:[{wtprs_left:.3g},{wtprs_right:.3g}]")

print("Unique values: ", [f"{i:.3g}" for i in 60 * simulatedValues['WTP RS time'].unique()])


def wtpForSubgroup(filter):
    sim = simulatedValues[filter]
    wtprs = (60 * sim['WTP RS time']).mean()
    wtprs_left = (60 * left[filter]['WTP RS time']).mean()
    wtprs_right = (60 * right[filter]['WTP RS time']).mean()
    return wtprs, wtprs_left,wtprs_right

filter = database.data['CS_1'] == 1
w,l,r = wtpForSubgroup(filter)
print(f"WTP car for PT people: {w:.3g} CI:[{l:.3g},{r:.3g}]")


