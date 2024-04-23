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
  modelName       = "MMNL_bestWorst",
  modelDescr      = "MMNL Best Worst model",
  indivID         = "ID", 
  analyticGrad    = TRUE,
  outputDirectory = "output",
  nCores = 4
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
library(reticulate)
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
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_pt                = 0,
              asc_cs                = 0,
              asc_rs                = 0,
             
              mu_log_b_tt           = 0.15,
              sigma_log_b_tt_inter  = 0,
              sigma_log_b_tt_intra  = 0,
              gamma_btt_shift       = 1,
              
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
              cost_income_elast     = 1,
              
              scale_best            = 1,
              scale_worst           = 1
              )

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_rs","scale_best","b_rsa","b_rsa_shift")

### Read in starting values for at least some parameters from existing model output file
apollo_beta = apollo_readBeta(apollo_beta, apollo_fixed, "MMNL_bestWorst", overwriteFixed=FALSE)

# ################################################################# #
#### DEFINE RANDOM COMPONENTS                                    ####
# ################################################################# #

### Set parameters for generating draws
apollo_draws = list(
  interDrawsType = "halton",
  interNDraws    = 50,
  interUnifDraws = c(),
  interNormDraws = c("draws_tt_inter"),
  intraDrawsType = "halton",
  intraNDraws    = 50,
  intraUnifDraws = c(),
  intraNormDraws = c("draws_tt_intra")
)

### Create random parameters
apollo_randCoeff = function(apollo_beta, apollo_inputs){
  randcoeff = list()
  

  
  randcoeff[["b_tt_value"]] =  mu_log_b_tt *(1 + sigma_log_b_tt_inter * draws_tt_inter + sigma_log_b_tt_intra * draws_tt_intra) * (gamma_btt_shift * mode_2023 + (1 - mode_2023))
  

  
  return(randcoeff)
}

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
  
  
  ### Compute probabilities for "best" choice using MNL model
  mnl_settings_best = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = 1,
    choiceVar    = choice_best,
    utilities    = list(pt  = scale_best*V[["pt"]],
                        cs  = scale_best*V[["cs"]],
                        rs  = scale_best*V[["rs"]]),
    componentName = "best"
  )
  P[["choice_best"]] = apollo_mnl(mnl_settings_best, functionality)
  
  ### Compute probabilities for "worst" choice using MNL model
  mnl_settings_worst = list(
    alternatives = c(pt=1, cs=2, rs=3),
    avail        = list(pt=(choice_best!=1), cs=(choice_best!=2), rs=(choice_best!=3)),
    choiceVar    = choice_worst,
    utilities    = list(pt  = -scale_worst*V[["pt"]],
                        cs  = -scale_worst*V[["cs"]],
                        rs  = -scale_worst*V[["rs"]]),
    componentName = "worst"
  )
  
  P[["choice_worst"]] = apollo_mnl(mnl_settings_worst, functionality)
  
  ### Combined model
  P = apollo_combineModels(P, apollo_inputs, functionality)
  
  ### Average across intra-individual draws
  P = apollo_avgIntraDraws(P, apollo_inputs, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Average across inter-individual draws
  P = apollo_avgInterDraws(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

estimate_settings = list(#constraints = c("b_tt_cs > 0", "b_tt_rs > 0"),
                         writeIter = FALSE
                         )

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

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO FILE, using model name)               ----
# ----------------------------------------------------------------- #






unconditionals <- apollo_unconditionals(model,apollo_probabilities, apollo_inputs)

plot(density(as.vector(unconditionals[["b_tt_value"]])))

conditionals <- apollo_conditionals(model,apollo_probabilities, apollo_inputs)

plot(density((conditionals$post.mean)))



a = as.vector(unconditionals[["b_tt_value"]])
b = data.frame(a)
b['c']=1
library(ggplot2)
p = ggplot(b, aes(x=c,y=a)) + 
  geom_boxplot()
p







