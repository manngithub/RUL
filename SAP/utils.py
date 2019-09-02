"""
This mododules provides utility functions 
"""
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
from sklearn import decomposition
from matplotlib.mlab import PCA
import warnings 
warnings.filterwarnings('ignore')

def TimeOfEvent(X,machine = None, time_col=None):
    """
    Observing the number of cycles for particular machine, create a time-of-event column for the machine
    
    :type X: Pandas Dataframe
    :param X: Dataframe that contains the current time as cycle
    
    :type machine: string
    :param machine: Name of column in which unique value will carry a value of Time of Event
    
    :type time_col: string
    :param time_col: Names of the time series columns in which you want the function to process 
    
    :return: modified dataframe with time of event 
    """
    print '[Info] Generating a column \'time_of_event\' using maximum cycle number...'
    
    assert machine != None, 'column engine_id is not provided'
    assert time_col != None, 'column cycle is not provided'
    
    X['time_of_event']=np.nan
    for eng in X[machine].value_counts().index:
        X['time_of_event'][X[machine]==eng] = max(X[time_col][X[machine]==eng])
    return X

def TimeToEvent(X, time_col=None, event_col=None, dropTargetNaN=None):
    """
    Using the current time and the time of event, create a time-to-event column as training label
    
    :type X: Pandas Dataframe
    :param X: Dataframe that contains the event time and current time as cycle
    
    :type time_col: string
    :param time_col: Name of the time series columns in which you want the function to process 
    
    :type event_col: string
    :param event_col: Name of the time of event column that is used to process time-to-event
    
    :type dropTargetNaN: boolean (True/False)
    :param dropTargetNaN: Flag to identify weather or not target variable NaN rows are dropped from the dataframe 
    
    :return: modified dataframe with time of event converted to time to event
    """
    print '[Info] Generating \'time_to_event\' value for each engine at given cycle...'
    
    assert time_col != None, 'column cycle is not provided'
    assert event_col != None, 'time of event column is missing'
    
    X['time_to_event']=X[event_col]-X[time_col]
    if dropTargetNaN==True:
        X=X.dropna(subset=['time_to_event'],how='any')
    return X


def dataframe(df):
    """
    data frame properties
    
    :type df: Pandas Dataframe
    :param df: dataframe with columns as features and rows as samples
    """
    assert df.shape != (0,0), 'dataframe is empty'
    print 'Number of columns in dataframe: %d'%len(df.columns)
    print '\nColumns are:- ',
    for col in df.columns:
        print col,
    
    print '\n\nTop 5 lines in dataframe:'
    print df.head()
    
def stats_histogram(df,c):
    """
    plot histogram for every column of dataframe
    
    :type df: Pandas Dataframe
    :param df: dataframe with columns as features and rows as samples
    
    :type c: integer
    :param c: maximum number of histogram plots in any single row
    """
    print 'Histogram Plots of every column variable:'
    
    cols = df.columns
    ncols = len(df.columns)
    plt.figure(figsize=(14,20))
    if ncols % c == 0:
        hist_rows = ncols/c
    else:
        hist_rows = ncols/c + 1
    for i, col in enumerate(cols):
        plt.subplot(hist_rows,c,i+1)
        plt.hist(df[col])
        #plt.xlabel(col)
        plt.title(col)
        
def stats_boxplot(df):
    """
    boxplot of each column variable in dataframe
    
    :type df: Pandas Dataframe
    :param df: dataframe with columns as features and rows as samples
    """
    df.plot(kind='box', subplots=False, layout=(1,len(df.columns)), figsize = (15,4))
    plt.show()
    
def biplot(score,coeff,PCnum,subplot_num):
    """
    biplot of scores in principal component space
    
    :type score: numpy array
    :param score: scores of datapoints in PC space
    
    :type coeff: numpy array
    :param coeff: projection coefficients of original features along principal components
    
    :type PCnum: list
    :param PCnum: principal components for X and Y axis
    
    :type subplot_num: integer
    :param subplot_num: subplot number
    """
    xs = score[:,0]
    ys = score[:,1]
    
    ax = plt.subplot(110 + subplot_num)
    ax.scatter(xs,ys)
    ax.set_xlabel('PC%d'%PCnum[0], fontsize=15)
    ax.set_ylabel('PC%d'%PCnum[1], fontsize=15)
    ax.grid()
    plt.show()
        
def pca(df,cols):
    """
    principal component analysis and biplot
    
    :type df: Pandas Dataframe
    :param df: dataframe with columns as features and rows as samples
    
    :type cols: list
    :param cols: list of column variable in df to be analyzed
    """
    X = df[cols]
    # PCA components
    pca_temp = decomposition.PCA(n_components = 3)
    pca_temp.fit(X.as_matrix())
    print 'PC coverage: [',
    for pcc in pca_temp.explained_variance_ratio_:
        print '%.2f, '%pcc,
    print ']'
    # Biplot
    pca=PCA(X.as_matrix())
    plt.figure(figsize=(6,5))
    # pca.Y is projection of original data into PCA space
    # pca.Wt is projection of original axis into PCA space
    biplot(pca.Y[:,0:2],pca.Wt[:,0:2],[1,2],1)

def single_plot(xdata,ydata,xlabel,ylabel):
    """
    plot the x and y label data
    
    :type xdata: numpy array
    :param xdata: x-axis data
    
    :type ydata: numpy array
    :param ydata: y-axis data
    
    :type xlabel: str
    :param xlabel: x-axis label
    
    :type ylabel: str
    :param ylabel: y-axis label
    """
    plt.figure(figsize=(15,3))
    plt.plot(xdata,ydata)
    plt.xlabel(xlabel, fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    plt.show()
        
def plot_RUL(engine, X, engines_to_check, prediction, real, RULtraining, dt):
    """
    plot the remaining useful life (RUL): true, predicted and average
    
    :type engine: integer
    :param engine: engine id
    
    :type X: Pandas Dataframe
    :param X: Dataframe that contains engine id and cycle
    
    :type engines_to_check: list
    :param engines_to_check: list of engine ids to generate the test results
    
    :type prediction: list
    :param prediction: list of RUL predictions for every engine id listed in engines_to_check
    
    :type real: list
    :param real: list of actual RUL for every engine id listed in engines_to_check
    
    :type RULtraining: list
    :param RULtraining: list of RUL of all engine ids from training data
    
    :type dt: integer (e.g. 10,20,30 etc...)
    :param dt: span of cycles as a single input to be learn by the model 
    """
    plt.figure(figsize=(12,3))
    location = engines_to_check.index(engine)
    Yprediction = prediction[location]
    cycle = X[X['engine_id']==engine].cycle[dt-1:]
    Yreal = real[location]
    AverageRUL = range(0,-len(Yreal),-1) + np.mean(RULtraining) - dt + 1
    plt.plot(cycle,AverageRUL,'k.',label='Average time-to-failure')
    plt.plot(cycle,Yreal,'b-',label='True time-to-failure')
    plt.plot(cycle,Yprediction,'r-',label='Predicted time-to-failure')
    plt.title('Engine: %d' %engine)
    plt.xlabel('Cycle',fontsize=12)
    plt.legend(loc='top left')
    plt.show()   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        