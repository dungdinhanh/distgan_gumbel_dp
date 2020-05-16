import pandas as pd
import json
import numpy as np

binary_data = pd.read_csv('./data/fire/preprocess_2.csv')

syn_data = pd.read_csv('./data/fire/syn_data_100000.csv', names=list(binary_data.columns))

syn_data = syn_data.round(0).astype('int')

with open('./data/fire/2016-specs.json', 'r') as infile:
    spec = json.load(infile)

dtype = []
sdim = []
columns = []
for column in binary_data.columns:
    column_name, number = column.split('_')
    if(number.isdigit() and (int(number)==1)):
        print(column_name)
        columns.append(column_name)
        properties = spec[column_name]
        if(properties['type']=='enum'):
            sdim.append(properties['count'])
            dtype.append(1)
a, b = 0, 0
column = 0
index = {}
for i in range(len(sdim)):
    b = a + sdim[i]
    index[column] = range(a, b)
    a = b
    column +=1 

syn_data_1 = syn_data.as_matrix()

new_data = {}
original_data = []
for j in range(len(index)):
    new_data[j] = syn_data_1[:,index[j]]
    new_data_1 = []
    for data in new_data[j]:
        new_data_1.append(list(data).index(1))
    original_data.append(new_data_1)
original_data = np.array(original_data).T

fire_data = pd.DataFrame(original_data, columns = columns)
fire_data.to_csv('./data/fire/post_syn_data_60000.csv', index=False)
