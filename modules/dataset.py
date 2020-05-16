import math
import glob
import os, sys
import csv

from numpy.random import RandomState
import json

# for image io
from skimage import io
from skimage.transform import resize

import numpy as np
import pandas as pd
import tensorflow as tf

# for cifar10
py_version = sys.version_info[0]

if py_version < 3:
    import cPickle as pickle
else:
    import pickle

# for mnist
from tensorflow.examples.tutorials.mnist import input_data

# for cifar10
#import cPickle as pickle
import pickle

def shuffle( data, random_seed = 1):
    data_copy = np.copy(data).tolist()
    rand = RandomState(random_seed)
    rand.shuffle(data_copy)
    return np.array(data_copy)
    
# List all dir with specific name 
def list_dir(folder_dir, ext="png"):
    all_dir = sorted(glob.glob(folder_dir+"*."+ext), key=os.path.getmtime)
    return all_dir

def load_pickle(myfile):
    exists = os.path.exists(myfile)
    if exists:
        fo = open(myfile, 'rb')
        if py_version < 3:
            data = pickle.load(fo) #python 2
        else:
            data = pickle.load(fo, encoding='latin1') #python 3
        fo.close()
        return data
    else:
        print('*** %s is invalid' % (myfile))
        exit()
        
def unpickle(myfile):
    exists = os.path.exists(myfile)
    if exists:
        fo = open(myfile, 'rb')
        if py_version < 3:
            dict = pickle.load(fo) #python 2
        else:
            dict = pickle.load(fo, encoding='latin1') #python 3
        fo.close()
        return dict['data'], dict['labels']
    else:
        print('*** %s is invalid' % (myfile))
        exit()

def imread(path, is_grayscale=False):
    if (is_grayscale):
        img = io.imread(path, is_grayscale=True).astype(np.float)
    else:
        img = io.imread(path).astype(np.float)
    return np.array(img)  
        
def read_split_cardfraud(data_dir, filenames):
    data_file = data_dir + '/' + filenames
    real_data = []
    
    with open(data_file,'r') as file:
        csvf = csv.reader(file, delimiter =",")
        index = 0
        for row in csvf:
            if index == 0:
                pass
            else:
                data_row = row[1:]

                data_proc = []
                for i,entry in enumerate(data_row):
                    data_proc.append(float(entry))
                real_data.append(data_proc)
            index += 1
    
    real_data = shuffle(np.array(real_data))

    for dim in range(real_data.shape[1] - 1):
        min_v = min(real_data[:,dim])
        max_v = max(real_data[:,dim])
        range_v = max_v - min_v
        real_data[:,dim] = (real_data[:,dim] - np.ones_like(real_data[:,dim])*min_v)/(range_v)

    real_data_train = real_data[:142403]
    real_data_test  = real_data[-142403:]
    
    with open(data_file[:-4] + '_train.pickle', 'wb') as handle:
        pickle.dump(real_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(data_file[:-4] + '_test.pickle', 'wb') as handle:
        pickle.dump(real_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_split_uci_epileptic_seizure(data_dir, filenames):
    
    data_file = data_dir + '/' + filenames
    real_data = []
    
    #load dataets
    with open(data_file,'r') as file:
        csvf = csv.reader(file, delimiter =",")
        index = 0
        for row in csvf:
            if index == 0:
                pass
            else:
                data_row  = row[1:]
                data_proc = []
                for i,entry in enumerate(data_row):
                    data_proc.append(float(entry))
                real_data.append(data_proc)
            index += 1
    #shuffle real data        
    real_data = shuffle(np.array(real_data))

    #normalize real data
    for dim in range(real_data.shape[1] - 1):
        min_v = min(real_data[:,dim])
        max_v = max(real_data[:,dim])
        range_v = max_v - min_v
        real_data[:,dim] = (real_data[:,dim] - np.ones_like(real_data[:,dim])*min_v)/(range_v)

    # convert classes to binary labels
    real_data[:, -1] = (real_data[:, -1]==1.0)
    
    # split the data
    real_data_train  = real_data[:-(real_data.shape[0]/2)]
    real_data_test = real_data[-(real_data.shape[0]/2):]
    
    with open(data_file[:-4] + '_train.pickle', 'wb') as handle:
        pickle.dump(real_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(data_file[:-4] + '_test.pickle', 'wb') as handle:
        pickle.dump(real_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)        

def detect_categorical_value(data, sdim):
    print(np.shape(data))
    print(sdim)
    print('max = %f, min = %f' % (np.max(data), np.min(data)))
    n = np.shape(data)[0]
    val = []
    c   = 0
    #print('----- categorizing data ------')
    print('length: %d' % (len(val)))
    for i in range(n):
        if len(val) == 0:
            val.append(data[i])
            c = 1
            print('adding %d to val' % (data[i]))
            print(val)
            print('length: %d' % (len(val)))
        else:
            flag = 0
            for j in range(c):
                if data[i] != val[j]:
                    flag = 1
                    break
            if flag == 1 and c < sdim:
                c = c + 1
                val.append(data[i])
                print('adding %d to val' % (data[i]))
                print(val)
            elif flag == 1 and c > sdim:
                print('Invalid data range with respective to sdim: %d' % (data[i]))
                exit()
        #print('[dataset.py -- detect_categorical_value] c = {}'.format(c))
    val = np.sort(val)
    print('final val = {}'.format(val))
    print('----- categorizing data ------')
    return val          

def encoding_categorical(cat_val, data, sdim):
    # print(data)
    # print(cat_val)
    # exit()
    n  = np.shape(data)[0]
    nv = len(cat_val)
    data_onehot = []
    for i in range(n):
        tmp = np.zeros((1, sdim))
        for j in range(nv):
            if data[i] == cat_val[j]:
                tmp[:,j] = 1
                break
        data_onehot.append(tmp)
        #print('data = %d' % (data[i]))
        #print('data (one-hot) = {}' .format (tmp))
    data_onehot = np.squeeze(np.concatenate(data_onehot, axis=0))
    return data_onehot

def categorize_data(data, sdim):
    
    # detect and sort categorical value (sorted)
    val = detect_categorical_value(data, sdim)
    # encoding data into categorical
    data_onehot = encoding_categorical(val, data, sdim)
    return data_onehot

def load_fire_department(data_dir, filenames, _categorical_softmax_use = 1):
    bin_data = pd.read_csv('./data/fire/preprocess_2.csv')
    with open('./data/fire/2016-specs.json', 'r') as infile:
        spec = json.load(infile)
    dtype = []
    sdim = []
    for column in bin_data.columns:
        coulmn_name, number = column.split('_')
        if(number.isdigit() and (int(number)==1)):
            print(coulmn_name)
            properties = spec[coulmn_name]
            # print(properties)
            # exit()
            if(properties['type']=='enum'):
                sdim.append(properties['count'])
                dtype.append(1)
    real_data = bin_data.as_matrix()
    label = np.array([0]*305133)
    # print(sdim)
    # exit()
    return dtype, sdim, real_data, label

def load_fire_department_integer(data_dir, filenames):
    df = pd.read_csv(filenames)
    with open('data/fire_department/2016-specs.json', 'r') as infile:
        specs = json.load(infile)
    columns = df.columns
    new_df = df.copy()
    for column in columns:
        spec = specs[column]
        min_v = spec['min']
        max_v = spec['max']
        range_v = max_v - min_v
        real_data = new_df[column].values.astype('float')
        if(range_v!=0):
            real_data = (real_data - np.ones_like(real_data)*min_v)/(1.0*range_v)
        else:
            real_data = range_v
        new_df[column] = real_data
    data = new_df.values
    labels = np.array([0]*data.shape[0])    
    return data, labels     

def load_fire_department_categorical(data_dir, filenames, _categorical_softmax_use = 1):
    label = np.array([0]*305133)
    df = pd.read_csv(filenames)
    with open('data/fire_department/2016-specs.json', 'r') as infile:
        specs = json.load(infile)
    columns = df.columns
    new_df = df.copy()
    dtype = []
    sdim = []
    new_data = np.array([])
    for column in columns:
        spec = specs[column]
        data = df[column].values
        if(spec['type']=='enum'):
            dtype.append(1)
            sdim.append(spec['count'])
            data  = np.eye(spec['count'])[data]
        else:
            dtype.append(0)
        if(new_data.size==0):
            new_data = data
        else:
            new_data = np.concatenate((new_data, data), axis=1)
    # print(new_data.shape)
    # print(sum(sdim))
    # exit()
    return dtype, sdim, new_data, label


def get_cervical_cancer_categorical_info():
    #(int) Age 
    #(int) Number of sexual partners 
    #(int) First sexual intercourse (age) 
    #(int) Num of pregnancies 
    
    #(bool) Smokes 
    #(bool) Smokes (years) 
    #(bool) Smokes (packs/year) 
    #(bool) Hormonal Contraceptives 
    
    #(int) Hormonal Contraceptives (years) 
    #(bool) IUD 
    #(int) IUD (years) 
    #(bool) STDs 
    
    #(int) STDs (number) 
    #(bool) STDs:condylomatosis 
    #(bool) STDs:cervical condylomatosis 
    #(bool) STDs:vaginal condylomatosis 
    
    #(bool) STDs:vulvo-perineal condylomatosis 
    #(bool) STDs:syphilis 
    #(bool) STDs:pelvic inflammatory disease 
    #(bool) STDs:genital herpes 
    
    #(bool) STDs:molluscum contagiosum 
    #(bool) STDs:AIDS 
    #(bool) STDs:HIV 
    #(bool) STDs:Hepatitis B 
    
    #(bool) STDs:HPV 
    #(int) STDs: Number of diagnosis 
    #(int) STDs: Time since first diagnosis 
    #(int) STDs: Time since last diagnosis 
    
    #(bool) Dx:Cancer 
    #(bool) Dx:CIN 
    #(bool) Dx:HPV 
    #(bool) Dx 
    
    #(bool) Hinselmann: target variable 
    #(bool) Schiller: target variable 
    #(bool) Cytology: target variable 
    #(bool) Biopsy: target variable
        
    # categorical or not (#4, #7 is not [0,1] on old dataset)
    dtype = [0, 0, 0, 0, \
             1, 0, 0, 1, \
             0, 1, 0, 1, \
             0, 1, 0, 1, \
             1, 1, 1, 1, \
             1, 0, 1, 1, \
             1, 0, 0, 0, \
             0, 0, 1, 1, \
             1, 1, 1, 1]
             
    # dim for softmax
    sdim  = [1, 1, 1, 1, \
             2, 1, 1, 2, \
             1, 2, 1, 2, \
             1, 2, 1, 2, \
             2, 2, 2, 2, \
             2, 1, 2, 2, \
             2, 1, 1, 1, \
             1, 1, 2, 2, \
             2, 2, 2, 1]

    #dtype = [0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0, \
             #0, 0, 0, 0]
             
    ## dim for softmax
    #sdim  = [1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1, \
             #1, 1, 1, 1]
                          
    return dtype, sdim     

def load_cervical_cancer(data_dir, filenames, _categorical_softmax_use = 1):
    
    dtype, sdim = get_cervical_cancer_categorical_info()
    
    data_file = data_dir + '/' + filenames
    
    real_data = np.load(data_file)
    # if do not suffer, the performance is high (eg. auc = 0.96)
    real_data = shuffle(np.array(real_data))
        
    if _categorical_softmax_use == 0:

        for dim in range(real_data.shape[1] - 1):
            min_v = min(real_data[:,dim])
            max_v = max(real_data[:,dim])
            range_v = max_v - min_v
            if(range_v!=0):
                real_data[:,dim] = (real_data[:,dim] - np.ones_like(real_data[:,dim])*min_v)/(range_v)
            else:
                real_data[:,dim] = range_v

        real_data_train  = real_data[:-int(real_data.shape[0]/2)]
        real_data_test = real_data[-int(real_data.shape[0]/2):]

        with open(data_file[:-4] + '_train.pickle', 'wb') as handle:
            pickle.dump(real_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(data_file[:-4] + '_test.pickle', 'wb') as handle:
            pickle.dump(real_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    elif _categorical_softmax_use > 0:
        
        new_dim = sum(sdim)
        print('[dataset.py -- load_cervical_cancer] new dim = %d' % (new_dim))
        real_data_cat = np.zeros((real_data.shape[0], new_dim))
        pos = 0
        for dim in range(real_data.shape[1] - 1):
            if sdim[dim] > 1:
                # processing data
                print('----- categorizing data ------')
                print('feature %d' % (dim))
                real_data_cat[:,pos:pos+sdim[dim]] = categorize_data(real_data[:,dim],sdim[dim])
            else:
                min_v = min(real_data[:,dim])
                max_v = max(real_data[:,dim])
                range_v = max_v - min_v
                if(range_v!=0):
                    # normalize data
                    real_data_cat[:,pos:pos+sdim[dim]] = np.reshape((real_data[:,dim] - np.ones_like(real_data[:,dim])*min_v)/(range_v),(-1,1))
                else:
                    real_data_cat[:,pos:pos+sdim[dim]] = range_v
            print('current pos: %d' % (pos))       
            pos = pos + sdim[dim]
        
        print(np.shape(real_data_cat))
             
        real_data_cat[:,-1] = real_data[:,-1]
        real_data_train = real_data_cat[:-int(real_data_cat.shape[0]/2)]
        real_data_test  = real_data_cat[-int(real_data_cat.shape[0]/2):]
        
        print(np.shape(real_data_train))
        print(np.shape(real_data_test))

        with open(data_file[:-4] + '_train_categorical.pickle', 'wb') as handle:
            pickle.dump(real_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(data_file[:-4] + '_test_categorical.pickle', 'wb') as handle:
            pickle.dump(real_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)         
    
    return dtype, sdim
    
class Dataset(object):

    def __init__(self, name='mnist', source='./data/cervical_cancer2/', one_hot=False, batch_size = 64, seed = 0, categorical_softmax_use = 0):


        self.name            = name
        self.source          = source
        self.one_hot         = one_hot
        self.batch_size      = batch_size
        self.seed            = seed
        self._categorical_softmax_use = categorical_softmax_use
        
        np.random.seed(seed) # To make your "random" minibatches the same as ours

        self.count           = 0

        tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.

        if name == 'creditcardfraud':
            data_files  = 'creditcard.csv'
            train_pickle = self.source + data_files[:-4] + '_train.pickle'
            test_pickle  = self.source + data_files[:-4] + '_test.pickle'
            exists_train = os.path.exists(train_pickle)
            exists_test  = os.path.exists(test_pickle)
            if exists_train == False or exists_test == False:
                read_split_cardfraud(source, data_files)
            self.data_label = load_pickle(train_pickle)
            entire=True
            if(entire):
                self.data_label_1 = load_pickle(train_pickle)
                self.data_label_2 = load_pickle(test_pickle)
                self.data_label = np.concatenate([self.data_label_1, self.data_label_2], axis=0)
            self.data   = self.data_label[:,:-1]
            self.labels = self.data_label[:,-1]
            nor_idx = np.squeeze(np.where(self.labels == 0.))
            ano_idx = np.squeeze(np.where(self.labels == 1.))
            
            # good result with original distgan: 2 duplicates noise (0.0, 0.1)
            # good result with distgan dp: 3 duplicates noise (0.0, 0.2)
            n_dups    = 0
            max_range = 0.2
            
            #self.labels[nor_idx] = self.labels[nor_idx] + np.random.uniform(0.0, max_range, np.shape(self.labels[nor_idx])) # use it
            #self.labels[ano_idx] = self.labels[ano_idx] - np.random.uniform(0.0, max_range, np.shape(self.labels[ano_idx])) # do not use this
                        
            data_dup  = []
            label_dup = []
            
            for i in range(n_dups):
                for v in ano_idx:
                    data_dup.append(self.data[v])
                    #label_dup.append(self.labels[v] - np.random.uniform(0.0, max_range)) # use this is better for unbalance
                    label_dup.append(self.labels[v]) # if using this not good because cannot generate label 1
                    #print(self.data_label[v])
            
            if(n_dups!=0):
                self.data = np.concatenate([self.data, np.array(data_dup)], axis = 0)
                self.labels = np.concatenate([self.labels, np.array(label_dup)], axis = 0)
                
            self.labels = np.reshape(self.labels,(-1,1))
            print('data shape: {}'.format(np.shape(self.data)))
            print('labels shape: {}'.format(np.shape(self.labels)))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        elif name == 'uci_epileptic_seizure':
            data_files  = 'uci_epileptic_seizure.csv'
            train_pickle = self.source + data_files[:-4] + '_train.pickle'
            test_pickle  = self.source + data_files[:-4] + '_test.pickle'
            exists_train = os.path.exists(train_pickle)
            exists_test  = os.path.exists(test_pickle)
            if exists_train == False or exists_test == False:
                read_split_uci_epileptic_seizure(source, data_files)
            self.data_label = load_pickle(train_pickle)
            self.data   = self.data_label[:,:-1]
            self.labels = self.data_label[:,-1]
            nor_idx = np.squeeze(np.where(self.labels == 0.))
            ano_idx = np.squeeze(np.where(self.labels == 1.))
            
            # duplicate the anomaly data
            n_dups = 4 #good results (reported)

            #n_dups = int(len(nor_idx) / len(ano_idx) * 0.7) # duplicate about 70% of the nominal/abnormal ratio
            print('all_shape: {}'.format(np.shape(self.data_label)))
            print('duplicate: {}'.format(n_dups))
                        
            data_dup  = []
            label_dup = []
            
            for i in range(n_dups):
                for v in ano_idx:
                    data_dup.append(self.data[v])
                    label_dup.append(self.labels[v]) # if using this not good because cannot generate label 1
            
            self.data = np.concatenate([self.data, np.array(data_dup)], axis = 0)
            self.labels = np.concatenate([self.labels, np.array(label_dup)], axis = 0)
                
            self.labels = np.reshape(self.labels,(-1,1))
            print('data shape: {}'.format(np.shape(self.data)))
            print('labels shape: {}'.format(np.shape(self.labels)))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
            
        elif name == 'cervical_cancer':
            
            data_files = 'multi_impute.npy'
                        
            if self._categorical_softmax_use == 0:
                train_pickle = self.source + data_files[:-4] + '_train.pickle'
                test_pickle  = self.source + data_files[:-4] + '_test.pickle'
            else:
                train_pickle = self.source + data_files[:-4] + '_train_categorical.pickle'
                test_pickle  = self.source + data_files[:-4] + '_test_categorical.pickle'
            
            exists_train = os.path.exists(train_pickle)
            exists_test  = os.path.exists(test_pickle)
            
            if exists_train == False or exists_test == False:
                self.dtype, self.sdim = load_cervical_cancer(source, data_files, self._categorical_softmax_use)
            else:
                self.dtype, self.sdim = get_cervical_cancer_categorical_info()
                
            self.data_label = load_pickle(train_pickle)
            entire=True
            if(entire):
                self.data_label_1 = load_pickle(train_pickle)
                self.data_label_2 = load_pickle(test_pickle)
                self.data_label = np.concatenate([self.data_label_1, self.data_label_2], axis=0)
            self.data   = self.data_label[:,:-1]
            self.labels = self.data_label[:,-1]
            nor_idx = np.squeeze(np.where(self.labels == 0.))
            ano_idx = np.squeeze(np.where(self.labels > 0.0))
                        
            # duplicate the anomaly data
            n_dups = 0
            print('label_shape: {}'.format(np.shape(self.data_label)))
            print('duplicate: {}'.format(n_dups))
            
            data_dup  = []
            label_dup = []
            
            for i in range(n_dups):
                for v in ano_idx:
                    data_dup.append(self.data[v])
                    label_dup.append(self.labels[v]) # if using this not good because cannot generate label 1
            
            if(n_dups!=0):
                self.data = np.concatenate([self.data, np.array(data_dup)], axis = 0)
                self.labels = np.concatenate([self.labels, np.array(label_dup)], axis = 0)
                
            self.labels = np.reshape(self.labels,(-1,1))
            print('data shape (duplicated): {}'.format(np.shape(self.data)))
            print('labels shape (duplicated): {}'.format(np.shape(self.labels)))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        elif name == 'fire_department':
            if(self._categorical_softmax_use > 0):
                data_files = self.source
                self.dtype, self.sdim, self.data, self.labels = load_fire_department(source, data_files, self._categorical_softmax_use)
                print(len(self.dtype))
                print(len(self.sdim))
            else:
                data_file = self.source
                self.data = pd.read_csv(data_file).as_matrix()
                self.labels = np.array([0]*305133)
            self.labels = np.reshape(self.labels,(-1,1))
            print('data shape (duplicated): {}'.format(np.shape(self.data)))
            print('labels shape (duplicated): {}'.format(np.shape(self.labels)))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        elif name == 'fire_department_integer':
            data_files = self.source
            self.data, self.labels = load_fire_department_integer(source, data_files)
            self.labels = np.reshape(self.labels,(-1,1))            
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
        elif name == 'fire_department_categorical':
            data_files = self.source
            self.dtype, self.sdim, self.data, self.labels = load_fire_department_categorical(source, data_files, self._categorical_softmax_use)
            self.labels = np.reshape(self.labels,(-1,1))
            self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)


            
    def load_test_data(self):
        if self.name == 'creditcardfraud':
            data_files  = 'creditcard.csv'
        elif self.name == 'uci_epileptic_seizure':
            data_files  = 'uci_epileptic_seizure.csv'
        elif self.name == 'cervical_cancer':
            data_files  = 'multi_impute.npy'
        
        if self._categorical_softmax_use > 0:
            test_pickle = self.source + data_files[:-4] + '_test_categorical.pickle'
        else:
            test_pickle = self.source + data_files[:-4] + '_test.pickle'
        exists_test  = os.path.exists(test_pickle)
        if exists_test:
           data_label = load_pickle(test_pickle)
        return data_label

    def load_train_data(self):
        return self.data_label
        
    def data_size(self):
        return np.shape(self.data)[0]    

    def db_name(self):
        return self.name
        
    def db_source(self):
        return self.source

    def data_dim(self):
        if self.name == 'mnist' or self.name == 'mnist_anomaly':
            return 784  #28x28
        elif self.name == 'cifar10' or self.name == 'cifar10_mini100':
            return 3072 #32x32x3
        elif self.name == 'celeba':
            return 12288 #64x64x3
        elif self.name == 'stl10':
            return 6912 # 48x48x3
        elif self.name == 'creditcardfraud':
            return 30   # 1 x 30
        elif self.name == 'uci_epileptic_seizure':
            return 179
        elif self.name == 'cervical_cancer':
            # no soft max
            #return 36   
            # categorical softmax
            if self._categorical_softmax_use > 0:
               return 55
            else:
               return 36 
        elif self.name == 'fire_department':
            # no soft max
            #return 36   
            # categorical softmax
            if self._categorical_softmax_use > 0:
               return sum(self.sdim)
            else:
               return 93
        elif self.name == 'fire_department_integer':
            return self.data.shape[1]
        elif self.name == 'fire_department_categorical':
            return self.data.shape[1]
        else:
            print('data_dim is unknown.\n')

    def get_categorical_softmax_use(self):
        return self._categorical_softmax_use
        
    def get_categorical_info(self):
        return self.dtype, self.sdim    
       
    def data_shape(self):
        if self.name == 'creditcardfraud':
            return [1, 30]
        elif self.name == 'uci_epileptic_seizure':
            return [1, 179]    
        elif self.name == 'cervical_cancer':
            # categorical softmax
            if self._categorical_softmax_use > 0:
               return [1, 55]    
            else:
               # no soft max
               return [1, 36]
        elif self.name == 'fire_department':
            # categorical softmax
            if self._categorical_softmax_use > 0:
               return [1, sum(self.sdim)]    
            else:
               # no soft max
               return [1, 93]
        elif self.name == 'fire_department_integer':
            return [1, self.data.shape[1]]
        elif self.name == 'fire_department_categorical':
            return [1, self.data.shape[1]]
        else:
            print('data_shape is unknown.\n')
            
    def mb_size(self):
        return self.batch_size

    def next_batch(self):
        if self.name in ['creditcardfraud', 'uci_epileptic_seizure', 'cervical_cancer', 'fire_department_integer', 'fire_department_categorical']:
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
            batch = np.concatenate([self.minibatches[self.count], self.minilabels[self.count]],axis=0)
            if(self.name in ['fire_department_integer', 'fire_department_categorical']):
                return self.minibatches[self.count].T
            self.count = self.count + 1
            return batch.T
        elif self.name in ['celeba', 'stl10']:
            batch = self.random_mini_batches([], self.batch_size, self.seed)
            return batch

    def next_batch2(self):

        if self.name in ['creditcardfraud', 'uci_epileptic_seizure', 'cervical_cancer']: #or self.name == 'stl10':
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches, self.minilabels = self.random_mini_batches(self.data.T, self.labels.T, self.batch_size, self.seed)
            batch = self.minibatches[self.count]
            label = self.minilabels[self.count]
            self.count = self.count + 1
            return batch.T, label.T
                        
    # nsamples = -1, samples the whole dataset
    # otherwise: the number of samples as indicated        
    def generate_nsamples(self, nsamples):
        if nsamples == -1 or nsamples == np.shape(self.data)[0]:
           return self.data
        else:
           cnt = 0
           data = []
           while cnt < nsamples:
                X_mb = self.next_batch()
                data.append(X_mb)
                cnt = cnt + np.shape(X_mb)[0]
           data = np.concatenate(data, axis=0)
           return data[0:nsamples,:]
           
    def fixed_mini_batches(self, mini_batch_size = 64):
        
        if self.name in ['creditcardfraud', 'uci_epileptic_seizure', 'cervical_cancer', 'fire_department']: #or self.name == 'stl10':
            X = self.data
            Y = self.labels
            m = X.shape[0]
            mini_batches = []
            mini_labels  = []
            num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch = X[k * self.batch_size : (k+1) * self.batch_size, :]
                mini_label = Y[k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch)
                mini_labels.append(mini_label)

            # Handling the end case (last mini-batch < mini_batch_size)
            #if m % mini_batch_size != 0:
            #    mini_batch = X[num_complete_minibatches * self.batch_size : m, :]
            #    mini_label = Y[num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch)
            #    mini_labels.append(mini_label)
            
            return mini_batches, mini_labels
            
    # Random minibatches for training
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X)
        """
        
        if self.name in ['creditcardfraud', 'uci_epileptic_seizure', 'cervical_cancer', 'fire_department', 'fire_department_integer', 'fire_department_categorical']: #or self.name == 'stl10':
            m = X.shape[1]                 # number of training examples
            mini_batches = []
            mini_labels  = []
                            
            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation]

            # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
            num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_batches.append(mini_batch_X)
                mini_batch_Y = shuffled_Y[:, k * self.batch_size : (k+1) * self.batch_size]
                mini_labels.append(mini_batch_Y)
            
            # Handling the end case (last mini-batch < mini_batch_size)
            #if m % mini_batch_size != 0:
            #    mini_batch_X = shuffled_X[:, num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch_X)
            
            return mini_batches, mini_labels
