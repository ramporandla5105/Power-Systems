# Predecting energy while EV charging using public charging station.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import timedelta

# Importing the dataset
dataset = pd.read_csv('EV_CS.csv')


donechr_time = dataset.iloc[:, 4].values

def HOUR2SECS(x):
    return int(x[:2])*3600 + int(x[3:5])*60 + int(x[6:])

month_encoder1 = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}


for i in range(len(donechr_time)):
    if type(donechr_time[i]) == float:
        dataset.drop(i, axis = 0, inplace = True) 
    
time_start = dataset.iloc[:, 2].values
discon_time = dataset.iloc[:, 3].values
donechr_time = dataset.iloc[:, 4].values



a1 = [];a = [];a3 =[]
for i in range(27944):
     p = time_start[i].split(" ")
     p1 = discon_time[i].split(" ")
     p2 = donechr_time[i].split(" ")
     #p[4] = HOUR2SECS(p[4])   
     #p12[4] = HOUR2SECS(p12[4])
     #p11[4] = HOUR2SECS(p11[4])
     
#     p1[4], p1[1] = diff(p11[4], p12[4], int(p11[1]), int(p12[1]))
         
     a.append(np.array(p))
     a1.append(np.array(p1))
     a3.append(np.array(p2))


time_start = np.array(a); time_end1 = np.array(a1);time_end2 = np.array(a3)
time_start = time_start[:,:-1];time_end1 = time_end1[:, :-1];time_end2 = time_end2[:, :-1]

diff_hours = []
for i in range(len(time_end1)):
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    date1 = time_end2[i,3] + '-' +  month_encoder1[time_end2[i,2]] + '-' + time_end2[i,1] + ' ' +  time_end2[i,4] + '.000'
    date2 = time_start[i,3] + '-' +  month_encoder1[time_start[i,2]] + '-' + time_start[i,1] + ' ' +  time_start[i,4] + '.000'
    diff = datetime.datetime.strptime(date1, datetimeFormat)- datetime.datetime.strptime(date2, datetimeFormat)
   
    
    diff_hours.append(diff.total_seconds() /3600 )




dataset1 = dataset.iloc[:,:].values
#dataset1 = np.append(dataset1, range(27944))
#dataset = pd.DataFrame(dataset1)
for i in diff_hours:
    if i>71:
        
        x = diff_hours.index(i)
        dataset1 = np.delete(dataset1, x,  0)
        diff_hours.remove(i)
    elif i<.09:
        x = diff_hours.index(i)
        dataset1 = np.delete(dataset1, x, 0)
        diff_hours.remove(i)


dataset = pd.DataFrame(dataset1)


dataset = dataset.fillna(0)



time_start = dataset.iloc[:, 2].values
discon_time = dataset.iloc[:, 3].values
donechr_time = dataset.iloc[:, 4].values
power = dataset.iloc[:,5].values


a1 = [];a = [];a3 =[]
for i in range(27894):
     p = time_start[i].split(" ")
     p1 = discon_time[i].split(" ")
     p2 = donechr_time[i].split(" ")
     p[4] = HOUR2SECS(p[4])   
     p2[4] = HOUR2SECS(p2[4])
     p1[4] = HOUR2SECS(p1[4])
     p[0] = p[0][:-1];p1[0] = p1[0][:-1];p2[0] = p2[0][:-1]
     
       
     a.append(np.array(p))
     a1.append(np.array(p1))
     a3.append(np.array(p2))


time_start = np.array(a); time_discon = np.array(a1);time_donechr = np.array(a3)
time_start = time_start[:,:-1];time_discon = time_discon[:, :-1];time_donechr = time_donechr[:, :-1]


y = power 

X1 = dataset.iloc[:, 8].values # station IDs
X2 = dataset.iloc[:, 11].values # user ID

(user_uniq,user_counts ) = np.unique(X2, return_counts=True)

p_max = np.zeros(427)
p_min = np.array([100 for i in range(427)]).astype(float)


for i in range(27894):
    if X2[i] != 0 :
        
        a = X2[i]
        
        if p_max[np.where(user_uniq ==X2[i])] < y[i] :
            p_max[np.where(user_uniq ==X2[i])] = y[i]
        if p_min[np.where(user_uniq ==X2[i])] > y[i]:
           p_min[np.where(user_uniq ==X2[i])] = y[i]
            
for i in range(427):
    if p_min[i] ==100: 
        p_min[i] = 0
P_max = np.zeros(27894)
P_min = np.zeros(27894)
for i in range(27893):
        P_max[i] = p_max[np.where(user_uniq==X2[i])]
        P_min[i] = p_min[np.where(user_uniq==X2[i])]



H = time_start 
H = np.hstack((H, np.atleast_2d(X1).T))
H = np.hstack((H, np.atleast_2d(X2).T))
H = np.hstack((H, np.atleast_2d(P_max).T))
H = np.hstack((H, np.atleast_2d(P_min).T))



names = ['week', 'date', 'month', 'year', 'time1', 'station','user ID', 'Pmax', 'Pmin']
H = pd.DataFrame(H, columns=names)

#encoding categorical data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
week_encoder = ColumnTransformer([('week',OneHotEncoder(),[0])],remainder="passthrough")
H = week_encoder.fit_transform(H)

month_encoder = ColumnTransformer([('month',OneHotEncoder(),[8])],remainder="passthrough")
H = month_encoder.fit_transform(H)

station_encoder = ColumnTransformer([('stations',OneHotEncoder(),[22])],remainder="passthrough")
H = station_encoder.fit_transform(H)

H[:, 73:76] = H[:, 73:76].astype(int)


#splitting dataset into test and training sets

from sklearn.model_selection import  train_test_split
z_train,z_test,y_train,y_test = train_test_split(H,y,test_size = 0.12,random_state = 0)


#feature scaling

from sklearn.preprocessing import StandardScaler
sc_z = StandardScaler()
z_train = sc_z.fit_transform(z_train)
z_test = sc_z.transform(z_test)


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(z_train, y_train)

# Predicting a new result
y_pred = regressor.predict(z_test)

from sklearn.svm import SVR
regressor1 = SVR()
regressor1.fit(z_train, y_train)
y_pred1 = regressor1.predict(z_test)


        



####################
#errors
###################
import math

RMSE = 0
R_square = 0
R1 = 0
MAE = 0

y_mean = np.mean(y_test)

for i in range(3348):
    RMSE += (y_pred[i] - y_test[i])**2
    R_square += (y_test[i] - y_pred[i])**2
    R1 += (y_test[i] - y_mean)**2
    if y_test[i] >= y_pred[i]:
        MAE += y_test[i] - y_pred[i]
    else:
        MAE += y_pred[i] - y_test[i]
RMSE = math.sqrt(RMSE*(1/len(y_test)))
R_square = R_square/R1
MAE = MAE/len(y_test)

print('errors of random forest regresson')
#root mean square error
print(RMSE , "RMSE")

#coeffient of determination
print(R_square , "R^2")

#mean absolute error
print(MAE , 'MAE')

RMSE = 0
R_square = 0
R1 = 0
MAE = 0
############
#errors of SVR 
############

y_mean = np.mean(y_test)

for i in range(3348):
    RMSE += (y_pred1[i] - y_test[i])**2
    R_square += (y_test[i] - y_pred1[i])**2
    R1 += (y_test[i] - y_mean)**2
    if y_test[i] >= y_pred1[i]:
        MAE += y_test[i] - y_pred1[i]
    else:
        MAE += y_pred1[i] - y_test[i]
RMSE = math.sqrt(RMSE*(1/len(y_test)))
R_square = R_square/R1
MAE = MAE/len(y_test)
print('errors of SVM')
#root mean square error
print(RMSE , "RMSE")

#coeffient of determination
print(R_square , "R^2")

#mean absolute error
print(MAE , 'MAE')



#time_end is disconnected time  time_end[5] is done charging time

from tkinter import *



root = Tk()
root.geometry('550x550')
root.resizable(0,0)
root.title('POWER PREDICTOR')
root.config(bg ='seashell3')


date_var = StringVar()
time_var = StringVar()
user_ID_var = StringVar()
Max_P_var = StringVar()
Min_P_var = StringVar()
station_ID_var = StringVar()
week_var = StringVar()
power = StringVar()

def submit():
    week = week_var.get()
    date  = date_var.get()
    entry_time = time_var.get()
    user_ID = user_ID_var.get()
    Max_P = Max_P_var.get()
    station_ID = station_ID_var.get()
    Min_P = Min_P_var.get()
    date = date.split(':')
    H=[]
    H.append(week);
    H.extend(date);
    H.append(HOUR2SECS(entry_time));
    H.append(station_ID);H.append(user_ID);H.append(Max_P);H.append(Min_P)
    H = np.array(H)
    H.resize(1,9)
    H = week_encoder.transform(H)
    H = month_encoder.transform(H)
    H = station_encoder.transform(H)
    H[73:76] = H[73:76]
    y = regressor.predict(H)
    power.set(y)
        
        
        
        

Label(root, text = 'POWER PREDICTOR' , font='arial 20 bold', bg = 'seashell2').pack()


Entry(root, font = 'arial 10 bold', textvariable = power, bg ='antiquewhite2',width = 15).place(x=245, y = 500)
Label(root, text = 'ENERGY kWh', font=('calibre',10, 'bold')).place(x= 120,y =500)

Entry(root, font = 'arial 10 bold', textvariable = date_var, bg ='antiquewhite2',width = 14,).place(x=220, y = 50)
Label(root, text = 'date(D:M:Y)', font=('calibre',10, 'bold')).place(x= 25,y =50)

Entry(root, font = 'arial 10 bold', textvariable = week_var, bg ='antiquewhite2',width = 14,).place(x=220, y = 95)
Label(root, text = 'week', font=('calibre',10, 'bold')).place(x= 25,y =95)


Entry(root, font = 'arial 10 bold', textvariable = time_var , bg ='antiquewhite2',width = 14,).place(x= 220, y =140)
Label(root, text = 'Entry time(h:m:s)', font=('calibre',10, 'bold')).place(x= 25,y =140)

Entry(root, font = 'arial 10 bold', textvariable = user_ID_var, bg ='antiquewhite2',width = 14,).place(x=220, y = 185)
Label(root, text = 'user ID', font=('calibre',10, 'bold')).place(x= 25,y =185)

Entry(root, font = 'arial 10 bold', textvariable = Max_P_var, bg ='antiquewhite2',width = 14,).place(x=220, y = 230)
Label(root, text = 'Max charging of user ', font=('calibre',10, 'bold')).place(x= 25,y =230)

Entry(root, font = 'arial 10 bold', textvariable = Min_P_var, bg ='antiquewhite2',width = 14,).place(x=220, y = 275)
Label(root, text = 'Min charging of user', font=('calibre',10, 'bold')).place(x= 25,y =275)

Entry(root, font = 'arial 10 bold', textvariable = station_ID_var, bg ='antiquewhite2',width = 14,).place(x=220, y = 320)
Label(root, text = 'station ID', font=('calibre',10, 'bold')).place(x= 25,y =320)

Button(root,text = 'PREDICT', command = submit).place(x = 225, y = 400)

root.mainloop()