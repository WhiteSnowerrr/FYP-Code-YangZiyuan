

# 1 -----------------------------------------------------------------------


library(dplyr)
library(reticulate)
library(dfidx)
library(mlogit)
source_python('bior.py')

df <- bioRdata()[[1]]
dfrs <- bioRdata()[[2]]
dfpt <- bioRdata()[[3]]



# base --------------------------------------------------------------------


Df <- mlogit.data(df, shape = "wide", varying = 56:121, choice = "choice_best", 
                    id = 'id', drop.index = TRUE)
DfRS <- mlogit.data(dfrs, shape = "wide", varying = 56:121, choice = "choice_best", 
                  id = 'id', drop.index = TRUE)
DfPT <- mlogit.data(dfpt, shape = "wide", varying = 56:121, choice = "choice_best", 
                  id = 'id', drop.index = TRUE)



# ml <- mlogit(choice_best ~ cost_s + crowd_s + trans_s + share_s + disin_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, Df)
# summary(ml)
# mlrs <- mlogit(choice_best ~ cost_s + crowd_s + trans_s + share_s + disin_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, DfRS)
# summary(mlrs)
# mlpt <- mlogit(choice_best ~ cost_s + crowd_s + trans_s + share_s + disin_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, DfPT)
# summary(mlpt)



ml <- mlogit(choice_best ~ cost_s + crowd_s + trans_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, Df, reflevel = "publicTrans")
summary(ml)
mlrs <- mlogit(choice_best ~ cost_s + crowd_s + trans_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, DfRS, reflevel = "publicTrans")
summary(mlrs)
mlpt <- mlogit(choice_best ~ cost_s + crowd_s + trans_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, DfPT, reflevel = "publicTrans")
summary(mlpt)



# rol ---------------------------------------------------------------------



Df <- mlogit.data(df, shape = "wide", varying = 56:121, choice = "ch", 
                  id = 'id', drop.index = TRUE, ranked = TRUE)
DfRS <- mlogit.data(dfrs, shape = "wide", varying = 56:121, choice = "ch", 
                    id = 'id', drop.index = TRUE, ranked = TRUE)
DfPT <- mlogit.data(dfpt, shape = "wide", varying = 56:121, choice = "ch", 
                    id = 'id', drop.index = TRUE, ranked = TRUE)

ml <- mlogit(ch ~ cost_s + crowd_s + trans_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, Df, reflevel = "publicTrans")
summary(ml)
mlrs <- mlogit(ch ~ cost_s + crowd_s + trans_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, DfRS, reflevel = "publicTrans")
summary(mlrs)
mlpt <- mlogit(ch ~ cost_s + crowd_s + trans_s + tt_cs_s + tt_rs_s | age_s + qoffice_3_s | 0, DfPT, reflevel = "publicTrans")
summary(mlpt)

distribution(ml)



coef(ml)[-3]/coef(ml)[3]


predict(ml, Df, type="response")





