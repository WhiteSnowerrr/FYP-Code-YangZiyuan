
l2f=function(t){
  out = t
  qup = unname(quantile(t, 0.75))
  qlow = unname(quantile(t, 0.25))
  for (i in 1:length(t)){
    if (t[i] > qup){
      out[i] = 3
    }else if (t[i] < qlow){
      out[i] = 1
    }else{
      out[i] = 2
    }
  }
  return(out)}

l2f2=function(t){
  out = t
  te = sort(unique(t))
  for (i in 1:length(t)){
    out[i] = which(te==t[i])
  }
  return(out)}

RDataBase = function(){


library(reticulate)

source_python('bior.py')
database = apoRdata()[[1]]



database$mode_2023 = database$mode_2023 + 1 #1 pt, 2rs
database$wfh_now = database$wfh_now + 1 
database$license = database$license + 1
database$qhousehold_1 = database$qhousehold_1 + 1
database$qhousehold_2 = database$qhousehold_2 + 1
database$qhousehold_3 = database$qhousehold_3 + 1
database$qhousehold_5 = database$qhousehold_5 + 1
database[database$qfamily_2==10,'qfamily_2'] = 0
database$qfamily_2 = database$qfamily_2 + 1
database[database$qfamily_3==10,'qfamily_3'] = 0
database$qfamily_3 = database$qfamily_3 + 1
database$qfamily_4 = database$qfamily_4 + 2
database[database$qfamily_4==12,'qfamily_4'] = 1
database[database$qfamily_5==10,'qfamily_5'] = 0
database$qfamily_5 = database$qfamily_5 + 1
database$qfamily_5.1 = database$qfamily_5.1 + 2
database[database$qfamily_5.1==12,'qfamily_5.1'] = 1
database$qfamily_7 = database$qfamily_7 + 2
database[database$qfamily_7==12,'qfamily_7'] = 1
database$qoffice_1 = database$qoffice_1 + 1
database$qoffice_2 = database$qoffice_2 + 1
database$qoffice_3 = database$qoffice_3 + 1
database$qoffice_4 = database$qoffice_4 + 1
database$att_pt_1 = database$att_pt_1 + 2
database$att_pt_2 = database$att_pt_2 + 2
database$att_pt_3 = database$att_pt_3 + 2
database$att_pt_4 = database$att_pt_4 + 2
database$pt_experience_1 = database$pt_experience_1 + 2
database[database$pt_experience_1==12,'pt_experience_1'] = 1
database$pt_experience_2 = database$pt_experience_2 + 2
database[database$pt_experience_2==12,'pt_experience_2'] = 1
database$pt_experience_3 = database$pt_experience_3 + 2
database[database$pt_experience_3==12,'pt_experience_3'] = 1
database$rs_experience_1 = database$rs_experience_1 + 2
database[database$rs_experience_1==12,'rs_experience_1'] = 1
database$rs_experience_2 = database$rs_experience_2 + 2
database[database$rs_experience_2==12,'rs_experience_2'] = 1
database$rs_experience_3 = database$rs_experience_3 + 2
database[database$rs_experience_3==12,'rs_experience_3'] = 1

#database$income_1 = l2f(database$income_1)
#database$income_2 = l2f(database$income_2)
database$floor_area = l2f(database$floor_area)
database$time_living = 2023 - database$time_living

database[database$att_cs_1==10,'att_cs_1'] = 0
database$att_cs_1 = database$att_cs_1 + 1
database[database$att_cs_2==10,'att_cs_2'] = 0
database$att_cs_2 = database$att_cs_2 + 1
database[database$att_cs_3==10,'att_cs_3'] = 0
database$att_cs_3 = database$att_cs_3 + 1

database[database$att_shift_rs_1==10,'att_shift_rs_1'] = 0
database$att_shift_rs_1 = database$att_shift_rs_1 + 1
database[database$att_shift_rs_2==10,'att_shift_rs_2'] = 0
database$att_shift_rs_2 = database$att_shift_rs_2 + 1
database[database$att_shift_rs_3==10,'att_shift_rs_3'] = 0
database$att_shift_rs_3 = database$att_shift_rs_3 + 1

database$att_pt = 1
database[database$att_pt_1==3|database$att_pt_2==3|database$att_pt_3==3|database$att_pt_4==3,'att_pt'] = 2

database$att_shift_rs = 1
database[database$att_shift_rs_1==4|database$att_shift_rs_2==4|database$att_shift_rs_3==4,'att_shift_rs'] = 2

database$qfamily = 1
database[database$qfamily_2!=1|database$qfamily_3!=1|database$qfamily_5!=1,'qfamily'] = 2


database$private_car = 1
database[database$private_car_1_1==1&database$private_car_1_2==1,'private_car'] = 2
database[database$private_car_1_1==2&database$private_car_1_2==2,'private_car'] = 3
database[database$private_car_1_1==1&database$private_car_1_2==2,'private_car'] = 4

database$att_pt_5 = 1
database[database$att_pt_1==3|database$att_pt_2==3,'att_pt_5'] = 2

database$att_pt_6 = 1
database[database$att_pt_3==3|database$att_pt_4==3,'att_pt_6'] = 2

database[database$qfamily_3!=1,'qfamily_3'] = 2

#database[database$edu>2,'edu'] = 2
#database[database$cs_rs_1_1<=2,'cs_rs_1_1'] = 1
#database[database$cs_rs_1_1>=3,'cs_rs_1_1'] = 2

#database[database$cs_rs_1_2<=2,'cs_rs_1_2'] = 1
#database[database$cs_rs_1_2>=3,'cs_rs_1_2'] = 2

database$income_d = database$income_2-database$income_1
#database[database$income_d<0,'income_d'] = -1
#database[database$income_d>0,'income_d'] = 1

return(database)
}






