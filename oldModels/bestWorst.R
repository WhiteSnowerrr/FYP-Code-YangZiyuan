# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "MNL_bestWorst",
  modelDescr      = "Best Worst model",
  indivID         = "ID", 
  #outputDirectory = "output",
  nCores = parallel::detectCores() - 1
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
source_python('bior.py')
database = apoRdata()[[1]]

### Create new variable with average income
###database$mean_income = mean(database$income)
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
database$income_2_s = database$income_2/10000


# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_pt                = 0,
              asc_cs                = 0,
              asc_rs                = 0,
             
              b_tt_cs               = 0,
              b_tt_rs               = 0,
              
              b_cost                = 0,
              b_trans               = 0,
              b_crowd               = 0,
              b_share               = 0,
              
              b_pta                 = 0,
              b_csa                 = 0,
              b_rsa                 = 0,
              
              b_pto                 = 0,
              b_cso                 = 0,
              b_rso                 = 0,
              scale_best            = 1,
              scale_worst           = 1)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_pt","b_rsa","b_pto","scale_best")

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
  b_tt_cs_value   = b_tt_cs * commuting_days_s
  b_tt_rs_value   = b_tt_rs * commuting_days_s
  b_cost_value    = b_cost * (1 + 1/income_2_s)
  
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[["pt"]]  = asc_pt + b_cost_value * pt1_cost_s +                                       + b_pta * age_s + b_pto * qoffice_3_s + b_crowd * pt1_crowd_s + b_trans * pt1_trans_s
  V[["cs"]]  = asc_cs + b_cost_value * cs2_cost_s + b_tt_cs_value * (pt1_tt_s - cs2_tt_s) + b_csa * age_s + b_cso * qoffice_3_s
  V[["rs"]]  = asc_rs + b_cost_value * rs3_cost_s + b_tt_rs_value * (pt1_tt_s - rs3_tt_s) + b_rsa * age_s + b_rso * qoffice_3_s + b_share * rs3_share_s
  
  
  ### Define settings for MNL model component
  el_settings = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = 1,
    choiceVars   = list(choice_best,choice_worst),
    utilities    = V,
    scales       = list(scale_best,-scale_worst)
  )
  
  ### Compute exploded logit probabilities
  P[["model"]] = apollo_el(el_settings, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

apollo_modelOutput(model)

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
apollo_lrTest("MNL_SP", model)

# ----------------------------------------------------------------- #
#---- MODEL PREDICTIONS AND ELASTICITY CALCULATIONS              ----
# ----------------------------------------------------------------- #

### Use the estimated model to make predictions
predictions_base = apollo_prediction(model, apollo_probabilities, apollo_inputs, prediction_settings=list(runs=30))

### Now imagine the cost for rail increases by 1%
database$cost_rail = 1.01*database$cost_rail

### Rerun predictions with the new data
apollo_inputs = apollo_validateInputs()
predictions_new = apollo_prediction(model, apollo_probabilities, apollo_inputs)

### Return to original data
database$cost_rail = 1/1.01*database$cost_rail
apollo_inputs = apollo_validateInputs()

### work with predictions at estimates
predictions_base=predictions_base[["at_estimates"]]

### Compute change in probabilities
change=(predictions_new-predictions_base)/predictions_base

### Not interested in chosen alternative now, so drop last column
change=change[,-ncol(change)]

### First two columns (change in ID and task) also not needed
change=change[,-c(1,2)]

### Look at first individual
change[database$ID==1,]

### And person 9, who has all 4 modes available
change[database$ID==9,]

### Summary of changes (possible presence of NAs for unavailable alternatives)
summary(change)

### Look at mean changes for subsets of the data, ignoring NAs
colMeans(change,na.rm=TRUE)
colMeans(subset(change,database$business==1),na.rm=TRUE)
colMeans(subset(change,database$business==0),na.rm=TRUE)
colMeans(subset(change,(database$income<quantile(database$income,0.25))),na.rm=TRUE)
colMeans(subset(change,(database$income>=quantile(database$income,0.25))|(database$income<=quantile(database$income,0.75))),na.rm=TRUE)
colMeans(subset(change,(database$income>quantile(database$income,0.75))),na.rm=TRUE)

### Compute own elasticity for rail:
log(sum(predictions_new[,6])/sum(predictions_base[,6]))/log(1.01)

### Compute cross-elasticities for other modes
log(sum(predictions_new[,3])/sum(predictions_base[,3]))/log(1.01)
log(sum(predictions_new[,4])/sum(predictions_base[,4]))/log(1.01)
log(sum(predictions_new[,5])/sum(predictions_base[,5]))/log(1.01)

# ----------------------------------------------------------------- #
#---- RECOVERY OF SHARES FOR ALTERNATIVES IN DATABASE            ----
# ----------------------------------------------------------------- #

sharesTest_settings = list()
sharesTest_settings[["alternatives"]] = c(car=1, bus=2, air=3, rail=4)
sharesTest_settings[["choiceVar"]]    = database$choice
sharesTest_settings[["subsamples"]]   = list(business=(database$business==1),
                                             leisure=(database$business==0))

apollo_sharesTest(model, apollo_probabilities, apollo_inputs, sharesTest_settings)

# ----------------------------------------------------------------- #
#---- MODEL PERFORMANCE IN SUBSETS OF DATABASE                   ----
# ----------------------------------------------------------------- #

fitsTest_settings = list()

fitsTest_settings[["subsamples"]] = list()
fitsTest_settings$subsamples[["business"]] = database$business==1
fitsTest_settings$subsamples[["leisure"]] = database$business==0

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











