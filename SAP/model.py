"""
This module contains functionalities of modeling and evaluation of LSTM models that are designed to estimate remaining
useful life of turbofan engine data.
"""
import numpy as np
import pandas as pd   
from SAP.utils import TimeOfEvent, TimeToEvent, stats_boxplot, single_plot
#import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
import warnings 
warnings.filterwarnings('ignore')

def LSTM_for_RUL(X, 
                 centering, 
                 normalization,
                 dt,
                 load_XYtraining_data,
                 model_parameters,
                 import_trained_model,
                 plot_training_history):
    """
    This function first prepare the training data for modeling. For modeling, it accepts model architecture.
    
    :type X: Pandas Dataframe
    :param X: Training dataframe; each row corresponds to an engine at specific cycle and its operational parameters and sensors measurements at that cycle
    
    :type centering: boolean (True/False)
    :param centering: - option to select whether input data to be min-max normalized and centered for better learning
    
    :type normalization: boolean (True/False)
    :param normalization: option to select whether output label data to be normalized
    
    :type dt: integer (e.g. 10,20,30 etc...)
    :param dt: span of cycles as a single input to be learn by the model 
    
    :type load_XYtraining_data: boolean (True/False)
    :param load_XYtraining_data: option to select whether to load already prepared training data that are available only for dt = 10, 20, 30, 40, 50
    
    :type model_parameters: dictionary 
    :param model_parameters: neural network architecture parameters
    
    :type import_trained_model: Keras Model Sequential
    :param import_trained_model: Keras trained model
    
    :type plot_training_history: boolean (True/False)
    :param plot_training_history: whether to plot model training error per epoch
    
    :return: tuple (training data dataframe, training input data, training label data, operational features, sensors features, cycle features, centering (True/False), normalization (True/False))    
    """
    
    # add a column 'time_of_event' to the dataframe 
    X = TimeOfEvent(X,machine = 'engine_id',time_col='cycle')
    
    # add a column 'time_to_event' to the dataframe
    X = TimeToEvent(X, time_col='cycle', event_col='time_of_event', dropTargetNaN=False)
    
    # initiate training data preparation
    data_training = X.copy()
    
    # centering - columns of settings and sensors measurements are min-max normalized and centered
    settings_min_max_mean, sensors_min_max_mean = [], []
    if centering:
        print('[Info] Centering the settings and sensors measurements...')
        # centering for settings columns
        settings = ['setting1','setting2','setting3']
        sensors = ['s%s'%i for i in range(1,22)]
        settings_min = X[settings].min()
        settings_max = X[settings].max()
        settings_mean = X[settings].mean()
        settings_min_max_mean = {'min':settings_min,'max':settings_max,'mean':settings_mean}
        for setting in settings:
            data_training[setting] = (X[setting] - settings_mean[setting])/(settings_max[setting] - settings_min[setting])
        
        # centering for sensors columns
        sensors_min = X[sensors].min()
        sensors_max = X[sensors].max()
        sensors_mean = X[sensors].mean()
        sensors_min_max_mean = {'min':sensors_min,'max':sensors_max,'mean':sensors_mean}
        for sensor in sensors:
            data_training[sensor] = (X[sensor] - sensors_mean[sensor])/(sensors_max[sensor] - sensors_min[sensor])
        # sensors box plot
        print('[Info] Box plot of column variables:')
        stats_boxplot(df = data_training[sensors])
        
        
    # normalization - column of output label 'time_to_event' is normalized
    time_of_event_max = []
    if normalization:
        print('[Info] Normalizing the remaining useful life at given cycle...')
        time_of_event_max = X['time_of_event'].max()
        data_training['time_to_event'] = X['time_to_event']/time_of_event_max
    
    # prepare training data - 'X' signifies input data and 'Y' signifies output label data
    Xtraining, Ytraining=[],[]
    if load_XYtraining_data: # this data is for dt = 10, 20, 30 ...
        select, ignore = os.path.split(os.path.abspath(__file__))
        Xtraining = np.load(select+'/Data/Xtraining_data_dt_'+str(dt)+'.npy')   
        Ytraining = np.load(select+'/Data/Ytraining_data_dt_'+str(dt)+'.npy')
    else:
        print('[Info] Preparing the input and output label data for model training...')
        Xtraining, Ytraining = [], []
        for engine in data_training['engine_id'].unique():
            cycle_max = data_training[data_training['engine_id'] == engine]['cycle'].max()
            for i in range(cycle_max - dt + 1):
                select_Xdata = data_training[data_training['engine_id'] == engine][settings+sensors][i:i+dt].as_matrix()
                Xtraining.append(select_Xdata)
                select_Ydata = data_training[data_training['engine_id'] == engine]['time_to_event'].iloc[i+dt-1]
                Ytraining.append(select_Ydata)
                
        Xtraining = np.array(Xtraining)
        Ytraining = np.array(Ytraining)
    
    # Model Development
    if import_trained_model: # trained models are available for dt = 10, 20, 30 ...
        select, ignore = os.path.split(os.path.abspath(__file__))
        model_path = select+'/Data/model_dt_'+str(dt)+'.h5'
        model = load_model(model_path)
    else:
        print('[Info] Model architecture defined...')
        # Design RNN Model Architecture
        lstm_layers = model_parameters['LSTM'] # number of LSTM layers and nodes in each layer
        ffnn_layers = model_parameters['FFNN'] # number of feed-forward neural network layers and nodes in each layer
        model = Sequential()
        model.add(LSTM(lstm_layers[0], return_sequences = True, input_shape = (dt,len(settings+sensors))))
        # add layers of LSTM
        if len(lstm_layers) > 1:
            for i in range(1,len(lstm_layers)):
                if (i == len(lstm_layers) - 1):
                    model.add(LSTM(lstm_layers[i]))
                else:
                    model.add(LSTM(lstm_layers[i]), return_sequences = True)
        # add layers of feed forward neural network
        for i in range(len(ffnn_layers)):
            model.add(Dense(ffnn_layers[i]))
        # network compilation
        model.compile(loss='mae',optimizer='adam')
        print('[Info] Model training in progress...')
        # model training
        model.fit(Xtraining,Ytraining,epochs=model_parameters['epoch'],batch_size=model_parameters['batch_size'],verbose = model_parameters['verbose'])
    
    # plot model training error
    if plot_training_history:
        print('[Info] Plot of model training error:')
        if import_trained_model:
            select, ignore = os.path.split(os.path.abspath(__file__))
            csv_file = select+'/Data/mae_loss_dt_'+str(dt)+'.csv'
            df = pd.read_csv(csv_file)
            single_plot(df.index,df,'epoch','mean absolute error')
    else:
        pass
        
    return (data_training, Xtraining, Ytraining, model, settings_min_max_mean, sensors_min_max_mean, time_of_event_max, centering, normalization, dt)

def Prepare_TestData(X, testing_RUL, settings_min_max_mean, 
                     sensors_min_max_mean, time_of_event_max, centering, 
                     normalization, dt, all_in_one):
    """
    This function prepares the testing data for model evaluation.
    
    :type X: Pandas Dataframe
    :param X: Test dataframe; each row corresponds to an engine at specific cycle and its operational parameters and sensors measurements at that cycle
    
    :type testing_RUL: Pandas Dataframe
    :param testing_RUL: remaining useful life for every engine (at the end of cycle) in the test data
    
    :type settings_min_max_mean: dictionary
    :param settings_min_max_mean: operational settings features
    
    :type sensors_min_max_mean: dictionary
    :param sensors_min_max_mean: sensor meaturements features
    
    :type time_of_event_max: float
    :param time_of_event_max: maximum value of cycle from any engine in training data
    
    :type centering: boolean (True/False)
    :param centering: - whether input data is min-max normalized and centered for better learning
    
    :type normalization: boolean (True/False)
    :param normalization: whether output label data is normalized
    
    :type dt: integer (e.g. 10,20,30 etc...)
    :param dt: span of cycles as a single input to be learn by the model 
    
    :type all_in_one: boolean (True/False)
    :param all_in_one: whether test data to be prepared separately for every engine or combined; separate option is better as it provides ease to understand the model evaluation results
    
    :return: tuple (Test input data, Test true label data, engines label for test input data, all_in_one to be used in any following functions)
    """
    # calculate 'time_to_event' column for testing data using given RUL 
    X['time_to_event'] = np.nan
    for engine in X['engine_id'].unique():
        cycle_max = X[X['engine_id'] == engine]['cycle'].max()
        RUL_at_end = testing_RUL[testing_RUL['engine_id'] == engine]['RUL']
        X.loc[X['engine_id'] == engine,'time_to_event'] = np.array(range(cycle_max-1+RUL_at_end,RUL_at_end-1,-1))
    
    # initiate testing data preparation
    data_testing = X.copy()
    
    if centering:
        print('[Info] Centering the input test data...')
        # centering for settings columns
        settings = ['setting1','setting2','setting3']
        settings_min = settings_min_max_mean['min']
        settings_max = settings_min_max_mean['max']
        settings_mean = settings_min_max_mean['mean']
        for setting in settings:
            data_testing[setting] = (X[setting] - settings_mean[setting])/(settings_max[setting] - settings_min[setting])
        
        # centering for sensors columns
        sensors = ['s%s'%i for i in range(1,22)]
        sensors_min = sensors_min_max_mean['min']
        sensors_max = sensors_min_max_mean['max']
        sensors_mean = sensors_min_max_mean['mean']
        for sensor in sensors:
            data_testing[sensor] = (X[sensor] - sensors_mean[sensor])/(sensors_max[sensor] - sensors_min[sensor])
    
    if normalization:
        print('[Info] Normalizing the remaining useful life of test data...')
        data_testing['time_to_event'] = X['time_to_event']/time_of_event_max
    
    # prepare testing data - 'X' signifies input data and 'Y' signifies output label data
    if all_in_one: # assimilate all the data regardless of engine_id
        Xtesting, Ytesting, engines = [], [], []
        for engine in data_testing['engine_id'].unique():
            cycle_max = data_testing[data_testing['engine_id'] == engine]['cycle'].max()
            if cycle_max > (dt+5): # at least 5 samples from the engine are expected
                for i in range(cycle_max - dt + 1):
                    select_Xdata = data_testing[data_testing['engine_id'] == engine][settings+sensors][i:i+dt].as_matrix()
                    Xtesting.append(select_Xdata)
                    select_Ydata = data_testing[data_testing['engine_id'] == engine]['time_to_event'].iloc[i+dt-1]
                    Ytesting.append(select_Ydata)
                    engines.append(engine)
        Xtesting = np.array(Xtesting)
        Ytesting = np.array(Ytesting)
        engines = np.arrary(engines)
    else: # assimilate test data for every engine_id. This option will help while evaluating RUL for specific engine_id
        print('[Info] Assimilating test data for every engine...')
        Xtesting, Ytesting, engines = [], [], []
        for engine in data_testing['engine_id'].unique():
            Xtest_engine, Ytest_engine = [], []
            cycle_max = data_testing[data_testing['engine_id'] == engine]['cycle'].max()
            if cycle_max > (dt+5): # at least 5 samples from the engine are expected
                for i in range(cycle_max - dt + 1):
                    select_Xdata = data_testing[data_testing['engine_id'] == engine][settings+sensors][i:i+dt].as_matrix()
                    Xtest_engine.append(select_Xdata)
                    select_Ydata = data_testing[data_testing['engine_id'] == engine]['time_to_event'].iloc[i+dt-1]
                    Ytest_engine.append(select_Ydata)
                Xtesting.append(np.array(Xtest_engine))
                Ytesting.append(np.array(Ytest_engine))
                engines.append(engine)
        Xtesting = np.array(Xtesting)
        Ytesting = np.array(Ytesting)
        engines = np.array(engines)
    return (Xtesting, Ytesting, engines, all_in_one)
    
def Evaluate_Model(Xtesting, Ytesting, engines, model, all_in_one, engines_to_check):
    """
    This function evaluate the trained model on the test data for specified engines
    
    :type Xtesting: numpy array
    :param Xtesting: test input data
    
    :type Ytesting: numpy array
    :param Ytesting: test label data
    
    :type engines: numpy array or list
    :param engines: array/list of engines that are collected in test data
    
    :type model: Keras Model Sequential
    :param model: trained model
    
    :type all_in_one: boolean (True/False)
    :param all_in_one: whether test data to be prepared separately for every engine or combined; separate option is better as it provides ease to understand the model evaluation results; combined option is not available 
    
    :type engines_to_check: list
    :param engines_to_check: list of test engines that are tested on trained model
    
    :return: tuple (prediction, real, engines_to_check)
    """
    all_in_one = False
    if all_in_one:
        # presently we don't have functionality for combined model
        pass
    else:
        print('[Info] collecting RUL predictions on test data for specified engines...')
        # make RUL prediction for each engine separately
        prediction, real = [], []
        for engine in engines_to_check:
            if type(engines) == list:
                location = engines.index(engine)
            else:
                location = engines.tolist().index(engine)
            Xtest_engine = Xtesting[location]
            prediction_engine = model.predict(Xtest_engine)
            prediction.append(prediction_engine.reshape(prediction_engine.shape[0]))
            real.append(Ytesting[location])
    
    return (prediction, real, engines_to_check)
            
def iNormalization(prediction, real, normalization, time_of_event_max):
    """
    This function inverse the normalization on test labels
    
    :type prediction: numpy array
    :param prediction: RUL prediction on test data
    
    :type real: numpy array
    :param real: real RUL provided for test data
    
    :type normalization: boolean (True/False)
    :param normalization: whether output label data is normalized
    
    :type time_of_event_max: float
    :param time_of_event_max: maximum value of cycle from any engine in training data
    
    :return: tuple (prediction, real) after inverse of normalization if present
    """
    if normalization:
        for index in range(len(real)): # for every engine
            prediction[index] = prediction[index]*time_of_event_max
            real[index] = real[index]*time_of_event_max
        
    return (prediction, real)

def training_data_to_testing(data_testing, dt, load_training_to_testing_data):
    """
    Training data can be used as test data and here we performs this function
    
    :type data_testing: Pandas Dataframe
    :param data_testing: training data dataframe that was used to generate train data from modeling
    
    :type dt: integer (e.g. 10,20,30 etc...)
    :param dt: span of cycles as a single input to be learn by the model 
    
    :type load_training_to_testing_data: boolean (True/False)
    :param load_training_to_testing_data: whether to load the training data in evaluation data format
    
    :return: tuple (test input data, test output labels, engine labels)
    """
    if load_training_to_testing_data:
        select, ignore = os.path.split(os.path.abspath(__file__))
        Xtraining_to_testing = np.load(select+'/Data/Xtraining_to_testing_dt_'+str(dt)+'.npy')   
        Ytraining_to_testing = np.load(select+'/Data/Ytraining_to_testing_dt_'+str(dt)+'.npy')
        engines_training_to_testing = np.load(select+'/Data/engines_training_to_testing_dt_'+str(dt)+'.npy')
    else:
        settings = ['setting1','setting2','setting3']
        sensors = ['s%s'%i for i in range(1,22)]
        Xtraining_to_testing, Ytraining_to_testing, engines_training_to_testing = [], [], []
        for engine in data_testing['engine_id'].unique():
            Xtest_engine, Ytest_engine = [], []
            cycle_max = data_testing[data_testing['engine_id'] == engine]['cycle'].max()
            if cycle_max > (dt+5): # at least 5 samples from the engine are expected
                for i in range(cycle_max - dt + 1):
                    select_Xdata = data_testing[data_testing['engine_id'] == engine][settings+sensors][i:i+dt].as_matrix()
                    Xtest_engine.append(select_Xdata)
                    select_Ydata = data_testing[data_testing['engine_id'] == engine]['time_to_event'].iloc[i+dt-1]
                    Ytest_engine.append(select_Ydata)
                Xtraining_to_testing.append(np.array(Xtest_engine))
                Ytraining_to_testing.append(np.array(Ytest_engine))
                engines_training_to_testing.append(engine)
        Xtraining_to_testing = np.array(Xtraining_to_testing)
        Ytraining_to_testing = np.array(Ytraining_to_testing)
        engines_training_to_testing = np.array(engines_training_to_testing)
    return (Xtraining_to_testing, Ytraining_to_testing, engines_training_to_testing)
    
