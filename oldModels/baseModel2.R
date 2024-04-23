# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)
library(reticulate)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "MNL_base",
  modelDescr      = "Basic MNL model",
  indivID         = "ID", 
  outputDirectory = "output",
  nCores = parallel::detectCores() - 1
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
source_python('bior.py')
database = apoRdata()[[1]]

### Create new variable with average income
database$mean_income = mean(database$income_2)
database$commuting_days_s = database$commuting_days*4
database$pt1_cost_s = database$pt1_cost/1000
database$cs2_cost_s = database$cs2_cost/1000
database$rs3_cost_s = database$rs3_cost/1000
database$pt1_trans_s = database$pt1_trans/1
database$pt1_crowd_s = database$pt1_crowd/10
database$rs3_share_s = database$rs3_share/10
database$pt1_tt_s = database$pt1_tt/1000
database$cs2_tt_s = database$cs2_tt/1000
database$rs3_tt_s = database$rs3_tt/1000
database$age_s = database$age/10
database$qoffice_3_s = database$qoffice_3/1


# ################################################################# #
#### ANALYSIS OF CHOICES                                         ####
# ################################################################# #

### Define settings for analysis of choice data to be conducted prior to model estimation
choiceAnalysis_settings <- list(
  alternatives = c(pt=1, cs=2, rs=3),
  avail        = 1,
  choiceVar    = database$choice_best,
  explanators  = database[,c("age","qoffice_3","income_2")],
  rows         = (database$mode_2023==1) #rs
)

### Run function to analyse choice data
apollo_choiceAnalysis(choiceAnalysis_settings, apollo_control, database)

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_pt                = 0,
              asc_cs                = 0,
              asc_rs                = 0,
             
              b_tt                  = 0,
              b_tt_shift            = 0,
              
              b_cost                = 0,
              b_cost_shift          = 0,
              b_trans               = 0,
              b_trans_shift         = 0,
              b_crowd               = 0,
              b_crowd_shift         = 0,
              b_share               = 0,
              b_share_shift         = 0,
              
              b_pta                 = 0,
              b_pta_shift           = 0,
              b_csa                 = 0,
              b_csa_shift           = 0,
              b_rsa                 = 0,
              b_rsa_shift           = 0,
              cost_income_elast     = 1
              
              )

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_rs","b_rsa","b_rsa_shift")

### Read in starting values for at least some parameters from existing model output file
#apollo_beta = apollo_readBeta(apollo_beta, apollo_fixed, "MNL_base", overwriteFixed=FALSE)

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  ### Create list of probabilities P
  P = list()
  
  ### Create alternative specific constants and coefficients using interactions with socio-demographics
  b_tt_value   = (b_tt + b_tt_shift * mode_2023) * commuting_days_s
  b_cost_value    = (b_cost + b_cost_shift * mode_2023) * (mean_income / income_2) ^ cost_income_elast
  b_pta_value     = b_pta + b_pta_shift * mode_2023
  b_csa_value     = b_csa + b_csa_shift * mode_2023
  b_rsa_value     = b_rsa + b_rsa_shift * mode_2023


  b_share_value   = b_share + b_share_shift * mode_2023
  b_crowd_value   = b_crowd + b_crowd_shift * mode_2023
  b_trans_value   = b_trans + b_trans_shift * mode_2023
  
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[["pt"]]  = asc_pt + b_cost_value * pt1_cost_s + b_tt_value * pt1_tt_s + b_pta_value * age_s + b_crowd_value * pt1_crowd_s + b_trans_value * pt1_trans_s
  V[["cs"]]  = asc_cs + b_cost_value * cs2_cost_s + b_tt_value * cs2_tt_s + b_csa_value * age_s
  V[["rs"]]  = asc_rs + b_cost_value * rs3_cost_s + b_tt_value * rs3_tt_s + b_rsa_value * age_s + b_share_value * rs3_share_s
  
  
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = 1,
    choiceVar    = choice_best,
    utilities    = V
  )
  
  ### Compute probabilities using MNL model
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

estimate_settings = list(#constraints = c("b_tt_cs > 0", "b_tt_rs > 0"),
                         writeIter = FALSE)

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs, estimate_settings)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

modelOutput_settings = list(printPVal=1
                            ,printClassical=FALSE
                            #,printOutliers=TRUE
                            )

apollo_modelOutput(model, modelOutput_settings)

predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
predictions_base=predictions_base[["at_estimates"]]
predictions_base[predictions_base$chosen>=predictions_base$pt & predictions_base$chosen>=predictions_base$cs & predictions_base$chosen>=predictions_base$rs, "chosen"]=1
predictions_base[predictions_base$chosen!=1, "chosen"]=0
sum(predictions_base$chosen)/nrow(predictions_base)

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO FILE, using model name)               ----
# ----------------------------------------------------------------- #

apollo_saveOutput(model)

# ################################################################# #
##### POST-PROCESSING                                            ####
# ################################################################# #

### Print outputs of additional diagnostics to new output file (remember to close file writing when complete)
apollo_sink()

# ----------------------------------------------------------------- #
#---- LR TEST AGAINST SIMPLE MNL MODEL                           ----
# ----------------------------------------------------------------- #

### Example syntax with both models loaded from file
apollo_lrTest("MNL_SP", "MNL_SP_covariates")

### Example syntax with one model in memory
apollo_lrTest("MNL_bestWorst", model)

# ----------------------------------------------------------------- #
#---- MODEL PREDICTIONS AND ELASTICITY CALCULATIONS              ----
# ----------------------------------------------------------------- #


### market share predictions
x="pt1_cost_s"
y=seq(-150,150,10)/1000
z='pt'
out=c()
for (i in y){
  predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

  database[,x] = database[,x]+i
  

  apollo_inputs = apollo_validateInputs()
  predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)

  database[,x] = database[,x]-i
  apollo_inputs = apollo_validateInputs()
  
  predictions_base=predictions_base[["at_estimates"]]
  

  change=(predictions_new-predictions_base)/predictions_base
  

  change=change[,-ncol(change)]
  

  change=change[,-c(1,2)]
  
  out = c(out,unname(colMeans(predictions_new,na.rm=TRUE)[z]))
}

library(ggplot2)

A = data.frame(x=y*1000,y=out)
xscale = max(A$x)-min(A$x)
yscale = max(A$y)-min(A$y)
la = toString(c(0,round(A[A$x==0, "y"],4)))
la = paste("(",la,")",sep="")
# Make the plot
ggplot() + 
  geom_line(data = A, aes(x=x, y=y),color = 'blue', size = 1) +
  geom_line(aes(x=c(0,0),y=c(0,A[A$x==0, "y"])), linetype = 'dashed') +
  geom_line(aes(x=c(min(A$x)*1.05,0),y=c(A[A$x==0, "y"],A[A$x==0, "y"])), linetype = 'dashed') +
  geom_point(aes(x=0,y=A[A$x==0, "y"]), color = 'red') + 
  annotate("text", x = 0+xscale*0.10, y = A[A$x==0, "y"]+yscale*0.20, label = la) + 
  labs(x = "Change in cost ($)", y = "Market share") +
  xlim(min(A$x)*1.05, max(A$x)*1.05) +
  ylim(0,max(A$y)*1.05) + 
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(), axis.line = element_line(colour = "black"))
# add the default price point on the line
#geom_vline(xintercept = 0.5, linetype = "dashed", color = "red") +




### revenue predictions
x="rs3_cost_s"
y=seq(0.1,15,0.1)
z='rs'
out=c()
for (i in y){
  predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))
  
  database[,x] = database[,x]*i
  
  
  apollo_inputs = apollo_validateInputs()
  predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)
  revenue = predictions_new[,z]*database[,x]*1000
  database[,x] = database[,x]/i
  apollo_inputs = apollo_validateInputs()
  
  predictions_base=predictions_base[["at_estimates"]]
  
  
  change=(predictions_new-predictions_base)/predictions_base
  
  
  change=change[,-ncol(change)]
  
  
  change=change[,-c(1,2)]
  
  out = c(out,mean(revenue,na.rm=TRUE))

}

library(ggplot2)

s = out[y==1]
out = out/s

A = data.frame(x=y,y=out)
xscale = max(A$x)-min(A$x)
yscale = max(A$y)-min(A$y)
la = toString(c(round(A[A$y==max(A$y), "x"],2),round(A[A$y==max(A$y), "y"],2)))
la=paste('(',la,')',sep='')
# Make the plot
ggplot() + 
  geom_line(data = A, aes(x=x, y=y),color = 'blue', size = 1) +
  geom_vline(xintercept=1, linetype = 'dashed') +
  geom_point(aes(x=A[A$y==max(A$y), "x"],y=A[A$y==max(A$y), "y"]), color = 'red') + 
  annotate("text", x = A[A$y==max(A$y), "x"]+xscale*0.08, y = A[A$y==max(A$y), "y"]+yscale*0.05, label = la) + 
  labs(x = "Change in cost (scale)", y = "Revenue") +

  xlim(min(A$x) , max(A$x)*1.05) +

  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border = element_blank(), axis.line = element_line(colour = "black"))+
  coord_cartesian(clip = 'off',ylim=c(min(A$y),max(A$y)*1.05))+

  annotate('text',x=1,y=min(A$y)-yscale*0.1,label='x=1',color='black')




### elasticity predictions
x="pt1_cost_s"
z='pt'
z2='cs'
z3='rs'
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

database[,x] = database[,x]*1.01 ### Now imagine the cost for rail increases by 1%
apollo_inputs = apollo_validateInputs()
predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)
database[,x] = database[,x]/1.01
apollo_inputs = apollo_validateInputs()
predictions_base=predictions_base[["at_estimates"]]
log(sum(predictions_new[,z])/sum(predictions_base[,z]))/log(1.01) ### Compute own elasticity for rail:
log(sum(predictions_new[,z2])/sum(predictions_base[,z2]))/log(1.01) ### Compute cross-elasticities for other modes
log(sum(predictions_new[,z3])/sum(predictions_base[,z3]))/log(1.01)





### The value of travel time
model$betaStop['b_tt']/model$betaStop['b_cost']
(model$betaStop['b_tt']+model$betaStop['b_tt_shift'])/(model$betaStop['b_cost']+model$betaStop['b_cost_shift'])


### Use the estimated model to make predictions
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

### Now imagine the cost for rail increases by 1%
database$pt1_cost_s = 1.5*database$pt1_cost_s

### Rerun predictions with the new data
apollo_inputs = apollo_validateInputs()
predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)

### Return to original data
database$pt1_cost_s = 1/1.5*database$pt1_cost_s
apollo_inputs = apollo_validateInputs()

### work with predictions at estimates
predictions_base=predictions_base[["at_estimates"]]

### Compute change in probabilities
change=(predictions_new-predictions_base)/predictions_base

### Not interested in chosen alternative now, so drop last column
change=change[,-ncol(change)]

### First two columns (change in ID and task) also not needed
change=change[,-c(1,2)]

### Summary of changes (possible presence of NAs for unavailable alternatives)
summary(change)

### Look at mean changes for subsets of the data, ignoring NAs
colMeans(change,na.rm=TRUE)
colMeans(subset(change,database$mode_2023==0),na.rm=TRUE)
colMeans(subset(change,database$mode_2023==1),na.rm=TRUE)
colMeans(subset(change,(database$income_2<quantile(database$income_2,0.25))),na.rm=TRUE)
colMeans(subset(change,(database$income_2>=quantile(database$income_2,0.25))|(database$income_2<=quantile(database$income_2,0.75))),na.rm=TRUE)
colMeans(subset(change,(database$income_2>quantile(database$income_2,0.75))),na.rm=TRUE)

### Compute own elasticity for rail:
log(sum(predictions_new[,3])/sum(predictions_base[,3]))/log(1.5)

### Compute cross-elasticities for other modes
log(sum(predictions_new[,4])/sum(predictions_base[,4]))/log(1.5)
log(sum(predictions_new[,5])/sum(predictions_base[,5]))/log(1.5)

# ----------------------------------------------------------------- #
#---- RECOVERY OF SHARES FOR ALTERNATIVES IN DATABASE            ----
# ----------------------------------------------------------------- #

sharesTest_settings = list()
sharesTest_settings[["alternatives"]] = c(pt=1, cs=2, rs=3)
sharesTest_settings[["choiceVar"]]    = database$choice_best
sharesTest_settings[["subsamples"]]   = list(pt=(database$mode_2023==0),
                                             rs=(database$mode_2023==1))

apollo_sharesTest(model, apollo_probabilities, apollo_inputs, sharesTest_settings)

# ----------------------------------------------------------------- #
#---- MODEL PERFORMANCE IN SUBSETS OF DATABASE                   ----
# ----------------------------------------------------------------- #

fitsTest_settings = list()

fitsTest_settings[["subsamples"]] = list()
fitsTest_settings$subsamples[["pt"]] = database$mode_2023==0
fitsTest_settings$subsamples[["rs"]] = database$mode_2023==1

apollo_fitsTest(model,apollo_probabilities,apollo_inputs,fitsTest_settings)

# ----------------------------------------------------------------- #
#---- FUNCTIONS OF MODEL PARAMETERS                              ----
# ----------------------------------------------------------------- #

deltaMethod_settings=list(expression=c(VTT_car_min="b_tt_car/b_cost",
                                       VTT_car_hour="60*b_tt_car/b_cost",
                                       b_tt_diff_car_rail="b_tt_car-b_tt_rail"))
apollo_deltaMethod(model, deltaMethod_settings)

# ----------------------------------------------------------------- #
#---- switch off writing to file                                 ----
# ----------------------------------------------------------------- #

apollo_sink()











