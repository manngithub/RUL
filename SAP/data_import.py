"""
This module contains functions to deliver the data as requested.
"""
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore') # ignore warnings
import os

def get_C_MAPSS_Data(path, dataset):
    """
    Get turbofan engine C_MAPSS data for modeling purpose
    
    :type path: string
    :param path: data directory path relative to working file
    
    :type dataset: string
    :param dataset: selected dataset from four datasets
    
    :return: tuple (training data, testing data, RUL of test data)
    """
    # find the path of current working directory
    select, ignore = os.path.split(os.path.abspath(__file__))
    
    # training data
    print('[Info] Training Data Loading...')
    data_training_FD2 = pd.read_csv(select+'/'+path+'/train_'+dataset+'.txt',sep=" ", header=None)
    data_training_FD2.head()
    # columns defined for dataframe
    engine_cycle = ['engine_id','cycle']
    settings = ['setting1','setting2','setting3']
    sensors = ['s%s'%i for i in range(1,22)]
    nan_cols = ['NaN1','NaN2']
    data_training_FD2.columns = engine_cycle + settings + sensors + nan_cols
    data_training_FD2 = data_training_FD2.drop(['NaN1','NaN2'],axis=1)
    
    # test data
    print('[Info] Testing Data Loading...')
    data_testing_FD2 = pd.read_csv(select+'/'+path+'/test_'+dataset+'.txt',sep=" ", header=None)
    data_testing_FD2.columns = engine_cycle + settings + sensors + nan_cols
    data_testing_FD2 = data_testing_FD2.drop(['NaN1','NaN2'],axis=1)
    
    # RUL for test data
    print('[Info] Loading records of RUL on this testing data...')
    data_testing_RUL_FD2 = pd.read_csv(select+'/'+path+'/RUL_'+dataset+'.txt',sep=" ", header=None)
    data_testing_RUL_FD2.columns = ['RUL','nan_col']
    data_testing_RUL_FD2 = data_testing_RUL_FD2.drop(['nan_col'],axis=1)
    data_testing_RUL_FD2['engine_id'] = range(1,data_testing_RUL_FD2.shape[0]+1)
    
    return (data_training_FD2, data_testing_FD2, data_testing_RUL_FD2)

