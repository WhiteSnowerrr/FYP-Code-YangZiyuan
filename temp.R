

tryCatch(
  { source("next/RbaseNext.R") },
  warning = function(w) { source("RbaseNext.R") }
)
database = RDataBase(c(1)) # PT User
cor.test(database$income_2,database$edu,alternative="greater")


database = RDataBase(c(2)) # CS User
cor.test(database$income_2,database$edu,alternative="greater")







a1 = nrow(database[database$att_shift_cs_1==1,])
a2 = nrow(database[database$att_shift_cs_1==2,])
a3 = nrow(database[database$att_shift_cs_1==3,])



cor.test(database$att_pt_1,database$att_pt_2)




database = RDataBase(c(1))
nrow(database[database$att_pt_3==3&database$choice_best==1,])/nrow(database[database$att_pt_3==3,])
nrow(database[database$att_pt_3!=3&database$choice_best==1,])/nrow(database[database$att_pt_3!=3,])




#######
database = RDataBase(c(1))

### Create new variable with average income
database$pt1_tt_s    = database$pt1_tt
database$cs2_tt_s    = database$cs2_tt
database$rs3_tt_s    = database$rs3_tt
database$pt1_walk_s  = database$pt1_walk
database$pt1_wait_s  = database$pt1_wait
database$cs2_walk_s  = database$cs2_walk
database$rs3_wait_s  = database$rs3_wait
database$pt1_cost_s  = database$pt1_cost
database$cs2_cost_s  = database$cs2_cost
database$rs3_cost_s  = database$rs3_cost
database$pt1_trans_s = database$pt1_trans
database$pt1_crowd_s = database$pt1_crowd
database$cs2_disin_s = database$cs2_disin
database$rs3_share_s = database$rs3_share




for (i in 1:dim(database)[1]){
  database$min_tt[i] = min(database$pt1_tt[i],database$cs2_tt[i],database$rs3_tt[i])
  database$min_walk[i] = min(database$pt1_walk[i],database$cs2_walk[i])
  database$min_wait[i] = min(database$pt1_wait[i],database$rs3_wait[i])
  if (database$commuting_days[i]==2){
    database$commuting_days_s[i] = 0
  }
  else{
    database$commuting_days_s[i] = 1
  }
  if (database$att_pt_1[i]==3){database$att_pt_1_s[i] = 1}else{database$att_pt_1_s[i] = 0}
  if (database$att_pt_2[i]==3){database$att_pt_2_s[i] = 1}else{database$att_pt_2_s[i] = 0}
  if (database$att_pt_3[i]==3){database$att_pt_3_s[i] = 1}else{database$att_pt_3_s[i] = 0}
  if (database$att_pt_4[i]==3){database$att_pt_4_s[i] = 1}else{database$att_pt_4_s[i] = 0}
  if (database$income_2_s[i]<=3){database$income_2_s[i] = 0}else{database$income_2_s[i] = 1}
  if (database$age_s[i]<=3){database$age_s[i] = 0}else{database$age_s[i] = 1}
  if (database$edu[i]<=2){database$edu[i] = 0}else{database$edu[i] = 1}
  if (database$wfh_now[i]<=0){database$wfh_now_s[i] = 0}else{database$wfh_now_s[i] = 1}
}

database$gender=database$gender-1
#database$qoffice_3_s = database$qoffice_3/1
dsum=sum(database$d23_19)
database$d23_19_s = database$d23_19/dsum
database$mean_income = mean(database$income_2)

t=subset(database, select=c('license_s','income_2_s','d23_19_s','age_s','gender','att_pt_2_s','att_pt_3_s'))


res <- cor(t)
round(res, 2)




#######
database = RDataBase(c(2))

### Create new variable with average income
database$pt1_tt_s    = database$pt1_tt
database$cs2_tt_s    = database$cs2_tt
database$rs3_tt_s    = database$rs3_tt
database$pt1_walk_s  = database$pt1_walk
database$pt1_wait_s  = database$pt1_wait
database$cs2_walk_s  = database$cs2_walk
database$rs3_wait_s  = database$rs3_wait
database$pt1_cost_s  = database$pt1_cost
database$cs2_cost_s  = database$cs2_cost
database$rs3_cost_s  = database$rs3_cost
database$pt1_trans_s = database$pt1_trans
database$pt1_crowd_s = database$pt1_crowd
database$cs2_disin_s = database$cs2_disin
database$rs3_share_s = database$rs3_share




for (i in 1:dim(database)[1]){
  database$min_tt[i] = min(database$pt1_tt[i],database$cs2_tt[i],database$rs3_tt[i])
  database$min_walk[i] = min(database$pt1_walk[i],database$cs2_walk[i])
  database$min_wait[i] = min(database$pt1_wait[i],database$rs3_wait[i])
  if (database$commuting_days[i]==2){
    database$commuting_days_s[i] = 0
  }
  else{
    database$commuting_days_s[i] = 1
  }
  if (database$att_pt_1[i]==3){database$att_pt_1_s[i] = 1}else{database$att_pt_1_s[i] = 0}
  if (database$att_pt_2[i]==3){database$att_pt_2_s[i] = 1}else{database$att_pt_2_s[i] = 0}
  if (database$att_pt_3[i]==3){database$att_pt_3_s[i] = 1}else{database$att_pt_3_s[i] = 0}
  if (database$att_pt_4[i]==3){database$att_pt_4_s[i] = 1}else{database$att_pt_4_s[i] = 0}
  if (database$income_2_s[i]<=3){database$income_2_s[i] = 0}else{database$income_2_s[i] = 1}
  if (database$age_s[i]<=3){database$age_s[i] = 0}else{database$age_s[i] = 1}
  if (database$edu[i]<=2){database$edu[i] = 0}else{database$edu[i] = 1}
  if (database$wfh_now[i]<=0){database$wfh_now_s[i] = 0}else{database$wfh_now_s[i] = 1}
}

database$gender=database$gender-1
#database$qoffice_3_s = database$qoffice_3/1
dsum=sum(database$d23_19)
database$d23_19_s = database$d23_19/dsum
database$mean_income = mean(database$income_2)

t=subset(database, select=c('income_2_s','d23_19_s','age_s','gender','att_pt_2_s','att_pt_3_s'))


res <- cor(t)
round(res, 2)


data1 = database[database$d23_19==0,]

data2 = database[database$d23_19!=0,]


mean(data1$gender)
mean(data2$gender)

t=data2$wfh_now_s

# 使用 aggregate 计算每个饲料类型的数据个数
group_count <- aggregate(t, by = list(t), FUN = length)

# 显示结果
group_count/length(t)












