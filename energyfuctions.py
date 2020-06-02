
import pandas as pd 
import numpy as np
from sklearn.linear_model import RidgeCV



# convert the datatype
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def rmse(predictions, targets):

    differences = predictions - targets                       #the DIFFERENCEs.

    differences_squared = differences ** 2                    #the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)/ targets.mean()  #ROOT of ^/ MEAN of target

    return rmse_val


def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas


# utilize the RidgeCV to run a base line model 

def do_RCV_train(train, test, features):

    train_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))


    for m in set(train['Encoded-Type']):
        
        print(f"Meter {m}", end="") 
        
        # instantiate model
        model = RidgeCV(
            alphas=np.logspace(-10, 1, 25), 
            normalize=True,
        )    
        
        # fit model
        model.fit(
            X=train.loc[train['Encoded-Type']==m, features].values, 
            y=train.loc[train['Encoded-Type']==m, "target"].values
        )

        # make predictions 
        train_preds[train['Encoded-Type'] ==m] = model.predict(train.loc[train['Encoded-Type']==m, features].values)
        test_preds[test['Encoded-Type'] ==m]   = model.predict(test.loc[test['Encoded-Type']==m, features].values)
        
        # transform predictions
        train_preds[train_preds < 0] = 0
        train_preds[train['Encoded-Type']==m] = np.expm1(train_preds[train['Encoded-Type']==m]) * train[train['Encoded-Type']==m]['Area']
        
        test_preds[test_preds < 0] = 0 
        test_preds[test['Encoded-Type']==m] = np.expm1(test_preds[test['Encoded-Type']==m]) * test[test['Encoded-Type']==m]['Area']
        
        # evaluate model
        meter_rmsle = rmse(
            train_preds[train['Encoded-Type']==m],
            train.loc[train['Encoded-Type']==m, "Record"].values
        )
        
        print(f", training rmse={meter_rmsle:0.5f}, ", end="")

        meter_rmsle = rmse(
            test_preds[test['Encoded-Type']==m],
            test.loc[test['Encoded-Type']==m, "Record"].values
        )

        print(f", test rmse={meter_rmsle:0.5f} ")


    print(f"Overall training set rmse={rmse(train_preds, train.Record.values):0.5f}")

    print(f"Overall test set for target building rmse={rmse(test_preds, test.Record.values):0.5f}")

    return test_preds


import matplotlib.pyplot as plt

def draw_prediction_plot(test_preds, test, building_number):
    test['prediction'] = test_preds
    for m in range(2):
        test_type = test[test['Encoded-Type']==m]
        fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
        test_type[['Time', 'Record']].set_index('Time').resample('H').mean()['Record'].plot(ax=axes, label='Actual', alpha=0.8).set_ylabel('Meter reading', fontsize=14)
        test_type[['Time', 'prediction']].set_index('Time').resample('H').mean()['prediction'].plot(ax=axes, label='prediction', alpha=1).set_ylabel('Meter reading', fontsize=14)
        axes.set_title(f'Mean Meter reading Actual and Prediction for Type {m} in buiding {building_number}', fontsize=16)
        axes.legend()
    plt.show()
    return