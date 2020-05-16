import pandas as pd
import numpy as np

def check_boolean(column):
    for value in column:
        if(not np.isnan(value) and value!=0.0 and value!=1.0):
            print("**********************", value)
            return False
    return True

def check_int(column):
    for value in column:
        if(not np.isnan(value) and value!=int(value)):
            return False
    return True

def check_nan(column):
    for value in column:
        if(np.isnan(value)):
            return True
    return False

max_value = 10**10 
def checkminmax(column1, column2):
    # column1[np.isnan(column1)] = max_value
    # column2[np.isnan(column1)] = max_value
    min1 = min(column1)
    min2 = min(column2)
    max1 = max(column1)
    max2 = max(column2)
    if(min1==min2 and max1==max2):
        return True
    print(min1, min2, max1, max2)
    return False


df = pd.read_csv('./data/Raw_Carvical_Cancer/kag_risk_factors_cervical_cancer_nan.csv')
# print(df)

real_data = df.values

booleans = []
ints = []
floats = []
nans = []
nan_booleans = []
nan_ints = []
nan_floats = []


for dim in range(real_data.shape[1] - 1):
    column = real_data[:,dim]
    if(check_nan(column)):
        nans.append(dim)
        if(check_boolean(column)):
            nan_booleans.append(dim)
        elif(check_int(column)):
            nan_ints.append(dim)
        else:
            nan_floats.append(dim)
    if check_boolean(column):
        booleans.append(dim)
    elif check_int(column):
        ints.append(dim)
    else:
        floats.append(dim)

print("bool", booleans)
print("int", ints)
print("float", floats)
print("nan_bool", nan_booleans)
print("nan_int", nan_ints)
print("nan_float", nan_floats)
print("NAN", nans)


import pandas as pd
import numpy as np
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

import pickle



imp = IterativeImputer(max_iter=100, random_state=0)

X_multi = imp.fit_transform(real_data)


# X_multi = np.genfromtxt('./data/CervicalCancer/multi_impute.csv',delimiter=',')



#post_process
for dim in nan_booleans:
    X_multi[:,dim] = np.round(X_multi[:,dim])
    for i in range(len(X_multi[:,dim])):
        if X_multi[i,dim]>1:
            X_multi[i,dim]=1
        if X_multi[i,dim]<0:
            X_multi[i,dim]=0
    print("check boolean column %d" %(dim), check_boolean(X_multi[:,dim]))

print("***************************************")

for dim in nan_ints:
    X_multi[:,dim] = np.round(X_multi[:,dim])
    X_min = min(real_data[:,dim])
    X_max = max(real_data[:,dim])
    for i in range(len(X_multi[:,dim])):
        if X_multi[i,dim]<X_min:
            X_multi[i,dim]=X_min
        if X_multi[i,dim]>X_max:
            X_multi[i,dim]=X_max
    print("check minmax int column %d" %(dim), checkminmax(X_multi[:,dim], real_data[:,dim]), check_int(X_multi[:,dim]))

print("***************************************")

for dim in nan_floats:
    X_min = min(real_data[:,dim])
    X_max = max(real_data[:,dim])
    for i in range(len(X_multi[:,dim])):
        if X_multi[i,dim]<X_min:
            X_multi[i,dim]=X_min
        if X_multi[i,dim]>X_max:
            X_multi[i,dim]=X_max
    print("check minmax float column %d" %(dim), checkminmax(X_multi[:,dim], real_data[:,dim]))

#checked by hand
for dim in [26,27]:
    X_multi[:,dim] = np.round(X_multi[:,dim])
    print("Checked 26th 27th columns are int and in [1,22] since a lot of NAN")

np.save("./data/Raw_Carvical_Cancer/multi_impute.npy", X_multi)
np.savetxt('./data/Raw_Carvical_Cancer/multi_impute.csv', X_multi, delimiter=",")

