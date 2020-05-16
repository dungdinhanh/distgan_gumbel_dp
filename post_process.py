
import pandas as pd

binary_data = pd.read_csv('./data/fire/binary_data.csv')

syn_data = pd.read_csv('./data/fire/syn_data_30005.csv', header=None)

dtype = []
sdim = []
index_map = {}
count = 0
new_data = binary_data.copy()
syn_data = syn_data.round(0).astype('int')
for column in binary_data.columns:
    number = column[-1]
    if(number.isdigit()):
        dtype.append(1)
        sdim.append(2)
        index_map[column] = count+1
        count+=2
    else:
        dtype.append(0)
        sdim.append(1)
        index_map[column] = count
        count+=1
for column in new_data.columns:
    new_data[column] = syn_data[index_map[column]]

new_data.to_csv('./data/fire/syn_binary_30005.csv', index=False)



