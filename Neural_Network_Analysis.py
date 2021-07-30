import json

from keras import callbacks
from pandas.core.frame import DataFrame
from psf_lib.python_speech_features.python_speech_features.base import mfcc
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.legend import _get_legend_handles_
import statistics
import csv

# Path to json file that stores MFCCs and subject labels for each processed sample
SOFT_DATA_PATH_MFCC = str(Path.cwd()) + "/mfcc_data_soft.json"
HARD_DATA_PATH_MFCC = str(Path.cwd()) + "/mfcc_data_hard.json"


# Loads data from the json file and reshapes X_data(samples, 1, 208) and y_data(samples, 1)
# Input: JSON path
# Ouput: X(mfcc data), y(labels), session_lengths
def load_data_from_json(data_path, nr_classes): 

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays and reshapes them
    X = np.array(data['mfcc'])
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    y = np.array(data["labels"])
    y = keras.utils.to_categorical(y, nr_classes)

    session_lengths = np.array(data['session_lengths'])
    
    print("Data succesfully loaded!")

    return X, y, session_lengths


# ----- DATA HANDLING ------
        
# Takes in data and labels, and splits it into train, validation and test sets by percentage
# Input: Data, labels, whether to shuffle, % validatiion, % test
# Ouput: X_train, X_validation, X_test, y_train, y_validation, y_test
def prepare_datasets_percentsplit(X, y, shuffle_vars, validation_size=0.2, test_size=0.25,):

    # Create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle_vars)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, shuffle=shuffle_vars)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# Takes in data and labels, and splits it into train and test sets by session
# Input: Data, labels, session_lengths and test_session_index
# Ouput: X_train, X_validation, X_test, y_train, y_validation, y_test
def prepare_datasets_sessions(X, y, session_lengths, test_session_index=4, nr_subjects=5):

    session_lengths = list(session_lengths)

    subject_starting_index = 0
    start_test_index = subject_starting_index + sum(session_lengths[0][:test_session_index-1])
    end_test_index = start_test_index + session_lengths[0][test_session_index-1]
    end_subject_index = subject_starting_index + sum(session_lengths[0])

    # Testing to check correctly slicing
    ''' 
    print(session_lengths[0], 'Sum:', sum(session_lengths[0]))
    print('Subject start:', subject_starting_index)
    print('Test start:', start_test_index)
    print('Test end:', end_test_index)
    print('Subject end:', end_subject_index, '\n -------')
    '''
    if start_test_index == subject_starting_index:
        X_test = X[start_test_index:end_test_index]
        y_test = y[start_test_index:end_test_index]
        X_train = X[end_test_index:end_subject_index]
        y_train = y[end_test_index:end_subject_index]
        
    elif end_test_index == end_subject_index:
        #print(X[subject_starting_index:start_test_index].shape)
        X_train = X[subject_starting_index:start_test_index]
        y_train = y[subject_starting_index:start_test_index]
        X_test = X[start_test_index:end_test_index]
        #print(X[start_test_index:end_test_index].shape, '\n ---')
        y_test = y[start_test_index:end_test_index]
        
    else:
        X_train = X[subject_starting_index:start_test_index]
        y_train = y[subject_starting_index:start_test_index]
        X_test = X[start_test_index:end_test_index]
        y_test = y[start_test_index:end_test_index]
        X_train = np.concatenate((X_train, X[end_test_index:end_subject_index]))     
        y_train = np.concatenate((y_train, y[end_test_index:end_subject_index]))
    #print(X_train.shape, '\n -------')
    subject_starting_index = max(end_subject_index, end_test_index)

    for i in range(1, nr_subjects):
        start_test_index = subject_starting_index + sum(session_lengths[i][:test_session_index-1])
        end_test_index = start_test_index + session_lengths[i][test_session_index-1]
        end_subject_index = subject_starting_index + sum(session_lengths[i])
        
        # Testing to check correctly slicing
        '''
        print(session_lengths[i], 'Sum:', sum(session_lengths[i]))
        print('Subject start:', subject_starting_index)
        print('Test start:', start_test_index)
        print('Test end:', end_test_index)
        print('Subject end:', end_subject_index, '\n -------')
        '''
        if start_test_index == subject_starting_index:
            X_test  =   np.concatenate((X_test, X[start_test_index:end_test_index]))
            y_test  =   np.concatenate((y_test, y[start_test_index:end_test_index]))
            X_train =   np.concatenate((X_train, X[end_test_index:end_subject_index]))
            y_train =   np.concatenate((y_train, y[end_test_index:end_subject_index]))
           
        elif end_test_index == end_subject_index:
            #print(X[subject_starting_index:start_test_index].shape)
            X_train =   np.concatenate((X_train, X[subject_starting_index:start_test_index]))
            y_train =   np.concatenate((y_train, y[subject_starting_index:start_test_index]))
            #print(X[start_test_index:end_test_index].shape, '\n ---')
            X_test  =   np.concatenate((X_test, X[start_test_index:end_test_index]))    
            y_test  =   np.concatenate((y_test, y[start_test_index:end_test_index]))    
        else:
            X_train =   np.concatenate((X_train, X[subject_starting_index:start_test_index]))
            y_train =   np.concatenate((y_train, y[subject_starting_index:start_test_index]))
            X_test  =   np.concatenate((X_test, X[start_test_index:end_test_index]))
            y_test  =   np.concatenate((y_test, y[start_test_index:end_test_index]))    
            X_train =   np.concatenate((X_train, X[end_test_index:end_subject_index]))     
            y_train =   np.concatenate((y_train, y[end_test_index:end_subject_index]))
        #print(X_train.shape, '\n -------')
        subject_starting_index = max(end_subject_index, end_test_index)

    return X_train, X_test, y_train, y_test

# NOT FUNCTIONAL
def prepare_datasets_new(test_session_indexes, X, y, session_lengths, nr_subjects=5, nr_sessions=4):

    X_list = []
    y_list = []
   
    for session_i in range(nr_sessions):
        X_session_list = []
        y_session_list = []
        for subject_i in range(nr_subjects):
        
            session_data_X = X[0:session_lengths[subject_i][session_i]]
            session_data_y = y[0:session_lengths[subject_i][session_i]]
            if session_i > 0:
                start_index = X_list[session_i-1].shape[0]
                session_data_X = X[start_index : start_index + session_lengths[subject_i][session_i]]
                session_data_y = y[start_index : start_index + session_lengths[subject_i][session_i]]
            X_session_list.append(session_data_X)
            y_session_list.append(session_data_y)
        X_list.append(np.concatenate(X_session_list))
        y_list.append(np.concatenate(y_session_list))

    X_test = []
    y_test = []
    X_train = []
    y_train = []


    for i in range(nr_sessions):
        if i in test_session_indexes:
            X_test.append(X_list[i])
            y_test.append(y_list[i])
        else:
            X_train.append(X_list[i])
            y_train.append(y_list[i])
    
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    return X_train, X_test, y_train, y_test

# Trains the model 
# Input: Keras.model, batch_size, nr epochs, training, and validation data
# Ouput: History
def train( model, X_train, y_train, verbose, batch_size=64, epochs=30, 
            X_validation=None, y_validation=None):

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    #csv_path = str(Path.cwd()) + '/logs/{}/{}_train_log.csv'.format(MODEL_NAME, MODEL_NAME)
    #csv_logger = CSVLogger(csv_path, append=False)

    if X_validation.any():
        history = model.fit(X_train, 
                            y_train, 
                            validation_data=(X_validation, y_validation), 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            verbose=verbose)
    else:
        history = model.fit(X_train,  
                            y_train,  
                            batch_size=batch_size, 
                            epochs=epochs, 
                            verbose=verbose)
    return history


# Gives nr of datapoints for chosen session
# Input: session_lengths 2d-list, session_nr, nr of subjects
# Ouput: int(datapoints)
def get_nr_in_session(session_lengths:list, session_nr, nr_subjects=5):
    summ = 0
    for i in range(nr_subjects):
        summ += session_lengths[i][session_nr-1]
    return summ

# Prints session and training data 
# Input: None
# Ouput: None -> print
def print_session_train_data(X_train, X_test, y_train, y_test, session_lengths, session_nr):
    print(X_train.size)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print('Datapoints in session ' + str(session_nr) + ':', get_nr_in_session(session_lengths, session_nr))
    print('Should be remaining:', 2806 - get_nr_in_session(session_lengths, session_nr))

# Reshapes training og test data into batches NOT RELEVANT?
# Input: training, test data (and validation), batch_size
# Ouput: training, test data (and validation)
def batch_formatting(X_train, X_test, y_train, y_test, batch_size=64, nr_classes=5, X_validation=None, y_validation=None):
    
    train_splits = X_train.shape[0] // batch_size
    train_rest = X_train.shape[0] % batch_size
    test_splits = X_test.shape[0] // batch_size
    test_rest = X_test.shape[0] % batch_size
    
    X_train = X_train[:-train_rest]
    y_train = y_train[:-train_rest]
    X_test = X_test[:-test_rest]
    y_test = y_test[:-test_rest]
    
    X_train_batch = np.reshape(X_train, (batch_size, train_splits, 208))
    y_train_batch = np.reshape(y_train, (batch_size, train_splits, nr_classes))
    X_test_batch = np.reshape(X_test, (batch_size, test_splits, 208))
    y_test_batch = np.reshape(y_test, (batch_size, test_splits, nr_classes))

    if X_validation != None:
        val_splits = X_validation.shape[0] // batch_size
        val_rest = X_validation.shape[0] % batch_size
        X_validation = X_validation[:-val_rest]
        y_validation = y_validation[:-val_rest]
        X_val_batch = np.reshape(X_validation, (batch_size, val_splits, 208))
        y_val_batch = np.reshape(y_validation, (batch_size, val_splits))
        return X_train_batch, X_test_batch, y_train_batch, y_test_batch, X_val_batch, y_val_batch

    return X_train_batch, X_test_batch, y_train_batch, y_test_batch

# Retrieves data sets for each session as test set and evalutes. DOES USE prediction_csv_logger as default
# the average of networks trained om them
# Input: raw data, session_lengths list, total nr of sessions, batch_size, and nr of epochs 
# Ouput: tuple(cross validation average, list(result for each dataset(len=nr_sessions)))
def session_cross_validation(model_name:str, X, y, session_lengths, nr_sessions, log_to_csv=True, batch_size=64, epochs=30):
    session_training_results = []
    for i in range(nr_sessions):

        X_train_session, X_test_session, y_train_session, y_test_session = prepare_datasets_sessions(X, y, session_lengths, i)
    
        # Model:
        if model_name == 'LSTM':
            model = LSTM(input_shape=(1, 208))

        elif model_name == 'GRU':
            model = GRU(input_shape=(1, 208))

        elif model_name == 'CNN_1D':
            X_train_session = np.reshape(X_train_session, (X_train_session.shape[0], 208, 1))
            X_test_session = np.reshape(X_test_session, (X_test_session.shape[0], 208, 1))
            model = CNN_1D(input_shape=(208, 1))

        elif model_name == 'FFN':
            model = FFN(input_shape=(1, 208))

        else:
            raise Exception('Model not found')

        #model.summary()

        
        train(model, X_train_session, y_train_session, verbose=1, batch_size=batch_size, epochs=epochs)
        test_loss, test_acc = model.evaluate(X_test_session, y_test_session, verbose=2)
        session_training_results.append(test_acc)
        if log_to_csv:
            prediction_csv_logger(X_test_session, y_test_session, model_name, model, i)
        del model
        K.clear_session()
        #print('Session', i, 'as test data gives accuracy:', test_acc)

    average_result = statistics.mean((session_training_results))

    return average_result, session_training_results

# Retrieves data sets for each session as train set and evalutes on the others.
# the average of networks trained om them
# Input: raw data, session_lengths list, total nr of sessions, batch_size, and nr of epochs 
# Ouput: tuple(cross validation average, list(result for each dataset(len=nr_sessions)))
def inverse_session_cross_validation(model_name:str, X, y, session_lengths, nr_sessions, log_to_csv=True, batch_size=64, epochs=30):
    session_training_results = []
    for i in range(nr_sessions):

        X_test_session, X_train_session, y_test_session, y_train_session = prepare_datasets_sessions(X, y, session_lengths, i)
    
        # Model:
        if model_name == 'LSTM':
            model = LSTM(input_shape=(1, 208))

        elif model_name == 'GRU':
            model = GRU(input_shape=(1, 208))

        elif model_name == 'CNN_1D':
            X_train_session = np.reshape(X_train_session, (X_train_session.shape[0], 208, 1))
            X_test_session = np.reshape(X_test_session, (X_test_session.shape[0], 208, 1))
            model = CNN_1D(input_shape=(208, 1))

        elif model_name == 'FFN':
            model = FFN(input_shape=(1, 208))

        else:
            raise Exception('Model not found')

        train(model, X_train_session, y_train_session, verbose=1, batch_size=batch_size, epochs=epochs)
        test_loss, test_acc = model.evaluate(X_test_session, y_test_session, verbose=0)
        session_training_results.append(test_acc)
        if log_to_csv:
            custom_path = '/{}_train_session{}_log.csv'
            prediction_csv_logger(X_test_session, y_test_session, model_name, model, i, custom_path)
        del model
        K.clear_session()
        #print('Session', i, 'as test data gives accuracy:', test_acc)

    average_result = statistics.mean((session_training_results))

    return average_result, session_training_results

# Takes in test data and logs input data and the prediction from a model
# Input: raw data, session_lengths list, total nr of sessions, batch_size, and nr of epochs 
# Ouput: tuple(cross validation average, list(result for each dataset(len=nr_sessions)))
def prediction_csv_logger(X, y, model_name, model, session_nr, custom_path=None):
    
    csv_path = str(Path.cwd()) + '/logs/{}/{}_session{}_log.csv'.format(model_name, model_name, session_nr+1)
    if custom_path:
        path = str(Path.cwd()) + '/logs/{}' + custom_path
        csv_path = path.format(model_name, model_name, session_nr+1)

    layerOutput = model.predict(X, verbose=0)

    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['input', 'prediction', 'solution'])
        data = zip(X, layerOutput, y)
        writer.writerows(data)
        csv_file.close()

# Prints info about session data
# Input: session_lengths
# Output: None -> print
def get_session_info(session_lengths_soft, session_lengths_hard):
    print('Soft: {}\nHard: {}'.format(session_lengths_soft, session_lengths_hard))
    soft_avg_sess = np.average(list(np.average(x) for x in session_lengths_soft))
    soft_avg_sub = np.sum(list(np.average(x) for x in session_lengths_soft))
    hard_avg_sub = np.sum(list(np.average(x) for x in session_lengths_hard))
    hard_avg_sess = np.average(list(np.average(x) for x in session_lengths_hard))
    print('Avg session:', soft_avg_sess, hard_avg_sess)
    print('Avg sub:', soft_avg_sub, hard_avg_sub)

# Reduces the size of the train and test set with values [0.0, 1.0]
# Input: Data sets, how much to reduce train set, how much to reduce test set with 
# Output: Reduced data sets
def reduce_data_set_sizes(X_train, X_test, y_train, y_test, train_reduction=0.5, test_reduction=0, nr_subjects=5):
    
    X_train = np.array_split(X_train, nr_subjects)
    y_train = np.array_split(y_train, nr_subjects)
    X_test = np.array_split(X_test, nr_subjects)
    y_test = np.array_split(y_test, nr_subjects)
    
    train_keep = int(X_train[0].shape[0] * (1 - train_reduction))
    test_keep = int(X_test[0].shape[0] * (1 - test_reduction))

    for i in range(nr_subjects):
        #print(len(X_train[i]))
        X_train[i] = X_train[i][:train_keep]
        y_train[i] = y_train[i][:train_keep]
        X_test[i] = X_test[i][:test_keep]
        y_test[i] = y_test[i][:test_keep]
        #print(len(X_train[i]))
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_train, X_test, y_train, y_test

# ----- PLOTS ------

# Plots the training history with two subplots. First training and test accuracy, and then 
# loss with respect to epochs
# Input: History(from model.fit(...))
# Ouput: None -> plot
def plot_train_history(history, val_data=False):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    if val_data:
        axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    if val_data:
        axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

# Plots the training history of four networks inverse cross-validated (single trained)
# Input: data, nr of sessions in total, batch_size and epochs
# Ouput: None -> plot
def plot_comp_spread_single(X, y, session_lengths, nr_sessions, batch_size=64, epochs=30):

    history_dict = {'GRU': [],
                    'LSTM': [],
                    'FFN': [],
                    'CNN_1D': []}
    
    for i in range(nr_sessions):

        X_test_session, X_train_session, y_test_session, y_train_session = prepare_datasets_sessions(X, y, session_lengths, i)

        model_GRU = GRU(input_shape=(1, 208))
        GRU_h = train(model_GRU, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs)
        history_dict['GRU'].append(GRU_h)
        del model_GRU
        K.clear_session()

        model_LSTM = LSTM(input_shape=(1, 208))
        LSTM_h = train(model_LSTM, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs)
        history_dict['LSTM'].append(LSTM_h)
        del model_LSTM
        K.clear_session()

        model_FFN = FFN(input_shape=(1, 208)) 
        FFN_h = train(model_FFN, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs)
        history_dict['FFN'].append(FFN_h)
        del model_FFN
        K.clear_session()

        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        X_train_session = np.reshape(X_train_session, (X_train_session.shape[0], 208, 1))
        X_test_session = np.reshape(X_test_session, (X_test_session.shape[0], 208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs)
        history_dict['CNN_1D'].append(CNN_1D_h)
        del model_CNN_1D
        K.clear_session()
    
    # Logging data to CSV. Just copy, not implemented
    '''
    # Log data stream to CSV
    csv_path = str(Path.cwd()) + '/logs/Network_acc_comparison_single/comparison_acc_data.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['GRU_train_acc', 'LSTM_train_acc', 'FFN_train_acc', 'CNN_1D_train_acc', 'GRU_val_acc', 'LSTM_val_acc', 'FFN_val_acc', 'CNN_1D_val_acc'])
        data = zip(*history_dict.values(), *history_dict_val.values())
        writer.writerows(data)
        csv_file.close()
    
    # Log best results to CSV
    csv_path = str(Path.cwd()) + '/logs/Network_acc_comparison_single/comparison_best.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['GRU_train_acc', 'LSTM_train_acc', 'FFN_train_acc', 'CNN_1D_train_acc', 'GRU_val_acc', 'LSTM_val_acc', 'FFN_val_acc', 'CNN_1D_val_acc'])
        writer.writerow(    [np.max(history_dict.get('GRU_train')), np.max(history_dict.get('LSTM_train')), np.max(history_dict.get('FFN_train')), np.max(history_dict.get('CNN_1D_train')), 
                            np.max(history_dict_val.get('GRU_val')), np.max(history_dict_val.get('LSTM_val')), np.max(history_dict_val.get('FFN_val')), np.max(history_dict_val.get('CNN_1D_val'))] )
        csv_file.close()
    '''

    fig, axs = plt.subplots(2, 2, sharey=True)
    plt.ylim(0, 1)

    # GRU plot:
    axs[0, 0].plot(history_dict['GRU'][0].history["accuracy"])
    axs[0, 0].plot(history_dict['GRU'][1].history["accuracy"], 'tab:orange')
    axs[0, 0].plot(history_dict['GRU'][2].history["accuracy"], 'tab:green')
    axs[0, 0].plot(history_dict['GRU'][3].history["accuracy"], 'tab:red')
    axs[0, 0].set_title('GRU')
    # LSTM plot:
    axs[0, 1].plot(history_dict['LSTM'][0].history["accuracy"])
    axs[0, 1].plot(history_dict['LSTM'][1].history["accuracy"], 'tab:orange')
    axs[0, 1].plot(history_dict['LSTM'][2].history["accuracy"], 'tab:green')
    axs[0, 1].plot(history_dict['LSTM'][3].history["accuracy"], 'tab:red')
    axs[0, 1].set_title('LSTM')
    # FFN plot: 
    axs[1, 0].plot(history_dict['FFN'][0].history["accuracy"])
    axs[1, 0].plot(history_dict['FFN'][1].history["accuracy"], 'tab:orange')
    axs[1, 0].plot(history_dict['FFN'][2].history["accuracy"], 'tab:green')
    axs[1, 0].plot(history_dict['FFN'][3].history["accuracy"], 'tab:red')
    axs[1, 0].set_title('FFN')
    # CNN_1D plot:
    axs[1, 1].plot(history_dict['CNN_1D'][0].history["accuracy"])
    axs[1, 1].plot(history_dict['CNN_1D'][1].history["accuracy"], 'tab:orange')
    axs[1, 1].plot(history_dict['CNN_1D'][2].history["accuracy"], 'tab:green')
    axs[1, 1].plot(history_dict['CNN_1D'][3].history["accuracy"], 'tab:red')
    axs[1, 1].set_title('CNN_1D')

    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Accuracy')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    plt.show()
    

# Plots the average training history of four networks inverse cross-validated (single trained)
# Input: data, nr of sessions in total, batch_size and epochs
# Ouput: None -> plot
def plot_comp_accuracy_single(X, y, session_lengths, nr_sessions, batch_size=64, epochs=30):
    #'''
    history_dict = {'GRU_train': [],
                    'LSTM_train': [],
                    'FFN_train': [], 
                    'CNN_1D_train': []}
    history_dict_val = {'GRU_val': [],  
                        'LSTM_val': [],
                        'FFN_val': [], 
                        'CNN_1D_val': []}
    
    for i in range(nr_sessions):
        # Prepare data
        X_val_session, X_train_session, y_val_session, y_train_session = prepare_datasets_sessions(X, y, session_lengths, i)
        
        # GRU
        model_GRU = GRU(input_shape=(1, 208))
        GRU_h = train(model_GRU, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_session, y_validation=y_val_session)
        history_dict['GRU_train'].append(GRU_h.history['accuracy'])
        history_dict_val['GRU_val'].append(GRU_h.history['val_accuracy'])
        del model_GRU
        K.clear_session()
        
        # LSTM
        model_LSTM = LSTM(input_shape=(1, 208))
        LSTM_h = train(model_LSTM, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_session, y_validation=y_val_session)
        history_dict['LSTM_train'].append(LSTM_h.history['accuracy'])
        history_dict_val['LSTM_val'].append(LSTM_h.history['val_accuracy'])
        del model_LSTM
        K.clear_session()
        
        # FFN
        model_FFN = FFN(input_shape=(1, 208)) 
        FFN_h = train(model_FFN, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_session, y_validation=y_val_session)
        history_dict['FFN_train'].append(FFN_h.history['accuracy'])
        history_dict_val['FFN_val'].append(FFN_h.history['val_accuracy'])
        del model_FFN
        K.clear_session()
        
        # CNN_1D
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        X_train_session = np.reshape(X_train_session, (X_train_session.shape[0], 208, 1))
        X_val_session = np.reshape(X_val_session, (X_val_session.shape[0], 208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_session, y_train_session, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_session, y_validation=y_val_session)
        history_dict['CNN_1D_train'].append(CNN_1D_h.history['accuracy'])
        history_dict_val['CNN_1D_val'].append(CNN_1D_h.history['val_accuracy'])
        del model_CNN_1D
        K.clear_session()
    
    # Averaging out session training for each network
    for key in history_dict:
        history_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*history_dict[key])))
    for key in history_dict_val:
        history_dict_val[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*history_dict_val[key])))

    '''
    history_dict = {'GRU_train': [0.5, 0.8, 0.4, 0.8],
                    'LSTM_train': [0.5, 0.9, 0.3, 0.9],
                    'FFN_train': [0.75, 0.8, 0.2, 0.7], 
                    'CNN_1D_train': [0.8, 0.95, 0.1, 0.6]}
    history_dict_val = {'GRU_val': [0.5, 0.8, 0.4, 0.8],  
                        'LSTM_val': [0.5, 0.9, 0.4, 0.8],
                        'FFN_val': [0.75, 0.8, 0.4, 0.8], 
                        'CNN_1D_val': [0.8, 0.95, 0.4, 0.8]}
    #'''

    # Log data stream to CSV
    csv_path = str(Path.cwd()) + '/logs/Network_acc_comparison_single/comparison_acc_data.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['GRU_train_acc', 'LSTM_train_acc', 'FFN_train_acc', 'CNN_1D_train_acc', 'GRU_val_acc', 'LSTM_val_acc', 'FFN_val_acc', 'CNN_1D_val_acc'])
        data = zip(*history_dict.values(), *history_dict_val.values())
        writer.writerows(data)
        csv_file.close()
    
    # Log best results to CSV
    csv_path = str(Path.cwd()) + '/logs/Network_acc_comparison_single/comparison_best.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['GRU_train_acc', 'LSTM_train_acc', 'FFN_train_acc', 'CNN_1D_train_acc', 'GRU_val_acc', 'LSTM_val_acc', 'FFN_val_acc', 'CNN_1D_val_acc'])
        writer.writerow(    [np.max(history_dict.get('GRU_train')), np.max(history_dict.get('LSTM_train')), np.max(history_dict.get('FFN_train')), np.max(history_dict.get('CNN_1D_train')), 
                            np.max(history_dict_val.get('GRU_val')), np.max(history_dict_val.get('LSTM_val')), np.max(history_dict_val.get('FFN_val')), np.max(history_dict_val.get('CNN_1D_val'))] )
        csv_file.close()

    # Plot:
    fig, axs = plt.subplots(2, sharey=True)
    plt.ylim(0, 1)
    plt.subplots_adjust(hspace=1.0, top=0.85, bottom=0.15, right=0.75)
    fig.suptitle('Average accuracy with cross-session-training', fontsize=16)

    axs[0].plot(history_dict['CNN_1D_train'], ':', label='CNN_1D')
    axs[0].plot(history_dict['LSTM_train'], '--', label='LSTM')
    axs[0].plot(history_dict['GRU_train'], '-', label='GRU')
    axs[0].plot(history_dict['FFN_train'], '-.', label='FFN')
    axs[0].set_title('Training accuracy')
    
    axs[1].plot(history_dict_val['CNN_1D_val'], ':', label='CNN_1D')
    axs[1].plot(history_dict_val['LSTM_val'], '--', label='LSTM')
    axs[1].plot(history_dict_val['GRU_val'], '-', label='GRU')
    axs[1].plot(history_dict_val['FFN_val'], '-.', label='FFN')
    axs[1].set_title('Validation accuracy')
    
    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Accuracy')
    
    plt.legend(bbox_to_anchor=(1.05, 1.5), title='Models used\n', loc='center left')
    plt.style.use('seaborn-dark-palette') 
    plt.show()

# Plots training and validation history for CNN_1D network with SOFT and HARD data (single trained)
# Input: SOFT and HARD raw data, respective session_lengths, *details
# Output: None -> plot
def plot_comp_SoftHard_single(X_soft, y_soft, X_hard, y_hard, session_lengths_soft, session_lengths_hard, nr_sessions, batch_size=64, epochs=30):
    #'''
    train_dict = {'SOFT':[], 'HARD':[]}
    val_dict = {'SOFT':[], 'HARD':[]}
    
    for i in range(nr_sessions):
        # Prepare data
        X_val_soft, X_train_soft, y_val_soft, y_train_soft = prepare_datasets_sessions(X_soft, y_soft, session_lengths_soft, i)
        X_val_hard, X_train_hard, y_val_hard, y_train_hard = prepare_datasets_sessions(X_hard, y_hard, session_lengths_hard, i)
        X_train_soft = np.reshape(X_train_soft, (X_train_soft.shape[0], 208, 1))
        X_val_soft = np.reshape(X_val_soft, (X_val_soft.shape[0], 208, 1))
        X_train_hard = np.reshape(X_train_hard, (X_train_hard.shape[0], 208, 1))
        X_val_hard = np.reshape(X_val_hard, (X_val_hard.shape[0], 208, 1))
        
        # CNN_1D SOFT
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_soft, y_train_soft, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_soft, y_validation=y_val_soft)
        train_dict['SOFT'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['SOFT'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

        # CNN_1D HARD
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_hard, y_train_hard, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_hard, y_validation=y_val_hard)
        train_dict['HARD'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['HARD'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

    # Averaging out session training for each network
    for key in train_dict:
        train_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*train_dict[key])))
    for key in val_dict:
        val_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*val_dict[key])))


    '''
    train_dict = {'SOFT': [0.1, 0.7, 0.5, 0.69],
                'HARD': [0.55, 0.9, 0.3, 0.92]}
    val_dict = {'SOFT': [0.34, 0.85, 0.41, 0.74],  
                'HARD': [0.63, 0.99, 0.49, 0.88]}
    '''

    # Log data stream to CSV
    csv_path = str(Path.cwd()) + '/logs/Soft_hard_comparison_single/soft_hard_comparison_acc_data.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['soft_train_acc', 'hard_train_acc', 'soft_val_acc', 'hard_val_acc'])
        data = zip(*train_dict.values(), *val_dict.values())
        writer.writerows(data)
        csv_file.close()
    
    # Log best results to CSV
    csv_path = str(Path.cwd()) + '/logs/Soft_hard_comparison_single/soft_hard_comparison_best.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['soft_train_best', 'hard_train_best', 'soft_val_best', 'hard_val_best'])
        writer.writerow( [np.max(train_dict.get('SOFT')), np.max(train_dict.get('HARD')), np.max(val_dict.get('SOFT')), np.max(val_dict.get('HARD'))] )
        csv_file.close()

    # Plot:
    fig, axs = plt.subplots(2, sharey=True)
    plt.ylim(0, 1)
    plt.subplots_adjust(hspace=1.0, top=0.85, bottom=0.15, right=0.75)
    fig.suptitle('Model training (1x session) and validation (3x session) with Natural/Strong typing behavior', fontsize=16)

    axs[0].plot(train_dict['SOFT'], ':', label='CNN_1D Natural')
    axs[0].plot(train_dict['HARD'], '--', label='CNN_1D Strong')
    axs[0].set_title('Training accuracy')
    
    axs[1].plot(val_dict['SOFT'], ':', label='CNN_1D Natural')
    axs[1].plot(val_dict['HARD'], '--', label='CNN_1D Strong')
    axs[1].set_title('Validation accuracy')
    
    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Accuracy')
    
    plt.legend(bbox_to_anchor=(1.05, 1.5), title='Typing behavior evaluated\n', loc='center left')
    plt.style.use('seaborn-dark-palette') 
    plt.show()


# Plots training and validation history for CNN_1D network with SOFT and HARD data (three-session-trained)
# Input: SOFT and HARD raw data, respective session_lengths, *details
# Output: None -> plot
def plot_comp_SoftHard_3(X_soft, y_soft, X_hard, y_hard, session_lengths_soft, session_lengths_hard, nr_sessions, batch_size=64, epochs=30):
    #'''
    train_dict = {'SOFT':[], 'HARD':[]}
    val_dict = {'SOFT':[], 'HARD':[]}
    
    for i in range(nr_sessions):
        # Prepare data
        X_train_soft, X_val_soft, y_train_soft, y_val_soft = prepare_datasets_sessions(X_soft, y_soft, session_lengths_soft, i)
        X_train_hard, X_val_hard, y_train_hard, y_val_hard = prepare_datasets_sessions(X_hard, y_hard, session_lengths_hard, i)
        X_train_soft = np.reshape(X_train_soft, (X_train_soft.shape[0], 208, 1))
        X_val_soft = np.reshape(X_val_soft, (X_val_soft.shape[0], 208, 1))
        X_train_hard = np.reshape(X_train_hard, (X_train_hard.shape[0], 208, 1))
        X_val_hard = np.reshape(X_val_hard, (X_val_hard.shape[0], 208, 1))
        
        # CNN_1D SOFT
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_soft, y_train_soft, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_soft, y_validation=y_val_soft)
        train_dict['SOFT'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['SOFT'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

        # CNN_1D HARD
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_hard, y_train_hard, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_hard, y_validation=y_val_hard)
        train_dict['HARD'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['HARD'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

    # Averaging out session training for each network
    for key in train_dict:
        train_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*train_dict[key])))
    for key in val_dict:
        val_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*val_dict[key])))


    '''
    train_dict = {'SOFT': [0.1, 0.7, 0.5, 0.69],
                'HARD': [0.55, 0.9, 0.3, 0.92]}
    val_dict = {'SOFT': [0.34, 0.85, 0.41, 0.74],  
                'HARD': [0.63, 0.99, 0.49, 0.88]}
    '''

    # Log data stream to CSV
    csv_path = str(Path.cwd()) + '/logs/Soft_hard_comparison_3/soft_hard_comparison_acc_data.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['soft_train_acc', 'hard_train_acc', 'soft_val_acc', 'hard_val_acc'])
        data = zip(*train_dict.values(), *val_dict.values())
        writer.writerows(data)
        csv_file.close()
    
    # Log best results to CSV
    csv_path = str(Path.cwd()) + '/logs/Soft_hard_comparison_3/soft_hard_comparison_best.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['soft_train_best', 'hard_train_best', 'soft_val_best', 'hard_val_best'])
        writer.writerow( [np.max(train_dict.get('SOFT')), np.max(train_dict.get('HARD')), np.max(val_dict.get('SOFT')), np.max(val_dict.get('HARD'))] )
        csv_file.close()

    # Plot:
    fig, axs = plt.subplots(2, sharey=True)
    plt.ylim(0, 1)
    plt.subplots_adjust(hspace=1.0, top=0.85, bottom=0.15, right=0.75)
    fig.suptitle('Model training (3x session) and validation (1x session) with Natural/Strong typing behavior', fontsize=16)

    axs[0].plot(train_dict['SOFT'], ':', label='CNN_1D Natural')
    axs[0].plot(train_dict['HARD'], '--', label='CNN_1D Strong')
    axs[0].set_title('Training accuracy')
    
    axs[1].plot(val_dict['SOFT'], ':', label='CNN_1D Natural')
    axs[1].plot(val_dict['HARD'], '--', label='CNN_1D Strong')
    axs[1].set_title('Validation accuracy')
    
    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Accuracy')
    
    plt.legend(bbox_to_anchor=(1.05, 1.5), title='Typing behavior evaluated\n', loc='center left')
    plt.style.use('seaborn-dark-palette') 
    plt.show()


# Plots training and validation history for CNN_1D network with SOFT and HARD data (VAL, two data sets)
# Input: SOFT and HARD raw data, respective session_lengths, *details
# Output: None -> plot
def plot_comp_val_SoftHard(X_soft, y_soft, X_hard, y_hard, session_lengths_soft, session_lengths_hard, nr_sessions, batch_size=64, epochs=30):
    #'''
    #train_dict = {'SOFT':[], 'HARD':[], 'SOFT_1':[], 'HARD_1':[]}
    val_dict = {'SOFT':[], 'HARD':[], 'SOFT_1':[], 'HARD_1':[]}
    
    for i in range(nr_sessions):
        # Prepare data
        X_train_soft, X_val_soft, y_train_soft, y_val_soft = prepare_datasets_sessions(X_soft, y_soft, session_lengths_soft, i)
        X_train_hard, X_val_hard, y_train_hard, y_val_hard = prepare_datasets_sessions(X_hard, y_hard, session_lengths_hard, i)
        X_train_soft = np.reshape(X_train_soft, (X_train_soft.shape[0], 208, 1))
        X_val_soft = np.reshape(X_val_soft, (X_val_soft.shape[0], 208, 1))
        X_train_hard = np.reshape(X_train_hard, (X_train_hard.shape[0], 208, 1))
        X_val_hard = np.reshape(X_val_hard, (X_val_hard.shape[0], 208, 1))
        
        # CNN_1D SOFT
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_soft, y_train_soft, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_soft, y_validation=y_val_soft)
        #train_dict['SOFT'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['SOFT'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

        # CNN_1D HARD
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_train_hard, y_train_hard, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_val_hard, y_validation=y_val_hard)
        #train_dict['HARD'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['HARD'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

        # ------ Single:
        
        # CNN_1D SOFT
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_val_soft, y_val_soft, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_train_soft, y_validation=y_train_soft)
        #train_dict['SOFT_1'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['SOFT_1'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()

        # CNN_1D HARD
        model_CNN_1D = CNN_1D(input_shape=(208, 1))
        CNN_1D_h = train(model_CNN_1D, X_val_hard, y_val_hard, 1, batch_size=batch_size, epochs=epochs, 
                                                            X_validation=X_train_hard, y_validation=y_train_hard)
        #train_dict['HARD_1'].append(list(CNN_1D_h.history['accuracy']))
        val_dict['HARD_1'].append(list(CNN_1D_h.history['val_accuracy']))
        del model_CNN_1D
        K.clear_session()


    # Averaging out session training for each network
    #for key in train_dict:
    #    train_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*train_dict[key])))
    for key in val_dict:
        val_dict[key] = list(np.average([x, y, z, c]) for x, y, z, c in list(zip(*val_dict[key])))


    '''
    train_dict = {'SOFT': [0.1, 0.7, 0.5, 0.69],
                'HARD': [0.55, 0.9, 0.3, 0.92]}
    val_dict = {'SOFT': [0.34, 0.85, 0.41, 0.74],  
                'HARD': [0.63, 0.99, 0.49, 0.88]}
    '''
    '''
    # Log data stream to CSV
    csv_path = str(Path.cwd()) + '/logs/Soft_hard_comparison_3/soft_hard_comparison_acc_data.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['soft_train_acc', 'hard_train_acc', 'soft_val_acc', 'hard_val_acc'])
        data = zip(*train_dict.values(), *val_dict.values())
        writer.writerows(data)
        csv_file.close()
    
    # Log best results to CSV
    csv_path = str(Path.cwd()) + '/logs/Soft_hard_comparison_3/soft_hard_comparison_best.csv'
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['soft_train_best', 'hard_train_best', 'soft_val_best', 'hard_val_best'])
        writer.writerow( [np.max(train_dict.get('SOFT')), np.max(train_dict.get('HARD')), np.max(val_dict.get('SOFT')), np.max(val_dict.get('HARD'))] )
        csv_file.close()
    '''

    # Plot:
    fig, axs = plt.subplots(2, sharey=True)
    plt.ylim(0, 1)
    plt.subplots_adjust(hspace=1.0, top=0.85, bottom=0.15, right=0.75)
    fig.suptitle('Model training and validation with Natural/Strong typing behavior', fontsize=16)

    axs[0].plot(val_dict['SOFT'], ':', label='CNN_1D Natural')
    axs[0].plot(val_dict['HARD'], '--', label='CNN_1D Strong')
    axs[0].set_title('Validation accuracy (3 session training)')
    
    axs[1].plot(val_dict['SOFT_1'], ':', label='CNN_1D Natural')
    axs[1].plot(val_dict['HARD_1'], '--', label='CNN_1D Strong')
    axs[1].set_title('Validation accuracy (1 session training)')
    
    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Accuracy')
    
    plt.legend(bbox_to_anchor=(1.05, 1.5), title='Typing behavior evaluated\n', loc='center left')
    plt.style.use('seaborn-dark-palette') 
    plt.show()

# Plots training and validation history for CNN_1D network with SOFT and HARD data from CSV file
# Input: None -> CSV from path
# Output: None -> plot & CSV log
def plot_N_S_val_comp():

    df_3 = pd.read_csv('/Users/Markus/Prosjekter git/Slovakia 2021/logs/Soft_hard_comparison_3/soft_hard_comparison_acc_data.csv')[['soft_val_acc', 'hard_val_acc']]
    df_1 = pd.read_csv('/Users/Markus/Prosjekter git/Slovakia 2021/logs/Soft_hard_comparison_single/soft_hard_comparison_acc_data.csv')[['soft_val_acc', 'hard_val_acc']]

    df_3 = df_3.rename(columns={'soft_val_acc': 'natural_val_3', 'hard_val_acc': 'strong_val_3'})
    df_1 = df_1.rename(columns={'soft_val_acc': 'natural_val_1', 'hard_val_acc': 'strong_val_1'})
    comp_df = pd.concat([df_3, df_1], axis=1)
    comp_df.to_csv('logs/Natural_Strong_comp_comb/N_S_val_comp.csv')

    # Plot new N/S val comp:
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(13, 4))
    plt.ylim(0, 1)
    plt.subplots_adjust(hspace=1.0, top=0.85, bottom=0.15, right=0.75)
    fig.text(0.435, 0.03, 'Epochs', ha='center')
    fig.text(0.07, 0.5, 'Accuracy', va='center', rotation='vertical')

    axs[0].plot(df_3['soft_val_acc'], ':', label='CNN_1D Natural')
    axs[0].plot(df_3['hard_val_acc'], '--', label='CNN_1D Strong')
    axs[0].set_title('Validation accuracy (3 session training)')
    
    axs[1].plot(df_1['soft_val_acc'], ':', label='CNN_1D Natural')
    axs[1].plot(df_1['hard_val_acc'], '--', label='CNN_1D Strong')
    axs[1].set_title('Validation accuracy (1 session training)')
    
    #for ax in axs:
    #    ax.set_xlabel('Epochs')
    #    ax.set_ylabel('Accuracy')
    
    plt.legend(bbox_to_anchor=(1.75, 0.5), title='Typing behavior evaluated\n', loc='center right')
    plt.ylim(0.50, 1.00)
    plt.show()


# ----- MODELS ------

# Creates a keras.model with focus on LSTM layers
# Input: input shape, classes of classification
# Ouput: model:Keras.model
def LSTM(input_shape, nr_classes=5):

    model = keras.Sequential(name='LSTM_model')
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128), input_shape=input_shape, name='Bidirectional_LSTM'))
    model.add(keras.layers.Dense(128, activation='relu', activity_regularizer=l2(0.005), name='Dense_relu'))
    model.add(keras.layers.Dropout(0.3, name='Dropout'))
    # Output layer
    model.add(keras.layers.Dense(nr_classes, activation='softmax', name='Dense_relu_output'))

    return model

# Creates a keras.model with focus on GRU layers
# Input: input shape, classes of classification
# Ouput: model:Keras.model
def GRU(input_shape, nr_classes=5):

    model = keras.Sequential(name='GRU_model')
    model.add(keras.layers.Bidirectional(keras.layers.GRU(128), input_shape=input_shape, name='Bidirectional_GRU'))
    model.add(keras.layers.Dense(128, activation='relu', activity_regularizer=l2(0.005), name='Dense_relu'))
    model.add(keras.layers.Dropout(0.3, name='Dropout'))
    # Output layer:
    model.add(keras.layers.Dense(nr_classes, activation='softmax', name='Softmax'))

    return model

# Creates a keras.model with a basic feed-forward-network
# Input: input shape, classes of classification
# Ouput: model:Keras.model
def FFN(input_shape, nr_classes=5):

    model = keras.Sequential(name='FFN_model')
    model.add(keras.layers.Reshape((input_shape[-1],), input_shape=input_shape))
    model.add(keras.layers.Dense(256, activation='relu', input_shape=input_shape, name='Dense_relu_1'))
    model.add(keras.layers.Dense(128, activation='relu', activity_regularizer=l2(0.005), name='Dense_relu_2'))
    model.add(keras.layers.Dense(64, activation='relu', activity_regularizer=l2(0.005), name='Dense_relu_3'))
    model.add(keras.layers.Dropout(0.3, name='Dropout'))
    # Output layer:
    model.add(keras.layers.Dense(nr_classes, activation='softmax', name='Softmax'))

    return model

# Creates a keras.model with focus on Convulotion layers
# Input: input shape, classes of classification
# Ouput: model:Keras.model
def CNN_1D(input_shape, nr_classes=5):

    model = keras.Sequential(name='CNN_model')
    model.add(keras.layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling1D(pool_size=5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # Ouput layer
    model.add(keras.layers.Dense(nr_classes, activation='softmax', name='Softmax'))

    return model


if __name__ == "__main__":

    # ----- Load data ------
        # X.shape = (2806, 1, 208)
        # y.shape = (2806, nr_subjects)
        # session_lengths.shape = (nr_subjects, nr_sessions)
    #X_soft, y_soft, session_lengths_soft = load_data_from_json(SOFT_DATA_PATH_MFCC, nr_classes=5)
    #X_hard, y_hard, session_lengths_hard = load_data_from_json(HARD_DATA_PATH_MFCC, nr_classes=5)

    # Parameters:
    NR_SUBJECTS = 5
    NR_SESSIONS = 4
    BATCH_SIZE = 64
    EPOCHS = 30

    TEST_SESSION_NR = 4
    VERBOSE = 1
    MODEL_NAME = 'CNN_1D'
    LOG = False
    
    # ----- Get prepared data: train, validation, and test ------
        # X_train.shape = (2806-X_test, 1, 208)
        # X_test.shape = (X_test(from session nr. ?), 1, 208)
        # y_train.shape = (2806-y_test, nr_subjects)
        # y_test.shape = (y_test(from session nr. ?), nr_subjects)

    #X_val, X_train, y_val, y_train = prepare_datasets_sessions(X_soft, y_soft, session_lengths_soft, TEST_SESSION_NR)
    #X_train, X_val, y_train, y_val = reduce_data_set_sizes(X_train, X_val, y_train, y_val, train_reduction=0.5, test_reduction=0)
    #print(X_soft.shape, y_soft.shape)
    #X_train, X_val, y_train, y_val = prepare_datasets_new([0, 1], X_soft, y_soft, session_lengths_soft)
    #print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


    # ----- Make model ------
    #model_GRU = GRU(input_shape=(1, 208)) # (timestep, 13*16 MFCC coefficients)
    #model_LSTM = LSTM(input_shape=(1, 208)) # (timestep, 13*16 MFCC coefficients)
    #model_CNN_1D = CNN_1D(input_shape=(208, 1)) # (timestep, 13*16 MFCC coefficients)
    
    #model_GRU.summary()
    #model_LSTM.summary()
    #model_CNN_1D.summary()

    # ----- Train network ------
    #history_GRU = train(model_GRU, X_train, y_train, verbose=VERBOSE, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #history_LSTM = train(model_LSTM, X_train, y_train, verbose=VERBOSE, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #history_CNN_1D = train( model_CNN_1D, np.reshape(X_train, (X_train.shape[0], 208, 1)), 
    #                        y_train, X_validation=np.reshape(X_val, (X_val.shape[0], 208, 1)), y_validation=y_val, verbose=VERBOSE, 
    #                        batch_size=BATCH_SIZE, epochs=EPOCHS)
    

    # ----- Plot train accuracy/error -----
    #plot_train_history(history_CNN_1D, val_data=True)


    # ----- Evaluate model on test set ------

    #test_loss, test_acc = model_GRU.evaluate(X_test, y_test, verbose=VERBOSE)
    #print('\nTest accuracy GRU:', test_acc, '\n')
    #test_loss, test_acc = model_LSTM.evaluate(X_test, y_test, verbose=VERBOSE)
    #print('\nTest accuracy LSTM:', test_acc, '\n')
    #test_loss, test_acc = model_CNN_1D.evaluate(np.reshape(X_test, (X_test.shape[0], 208, 1)), y_test, verbose=0)
    #print('\nTest accuracy CNN_1D:', test_acc, '\n')
    

    # ----- Store test predictions in CSV ------
    #prediction_csv_logger(np.reshape(X_test, (X_test.shape[0], 208, 1)), y_test, MODEL_NAME, model_CNN_1D, TEST_SESSION_NR)
    


    '''
    # ----- Cross validation ------
    # Trained on three sessions, tested on one
    average_GRU = session_cross_validation('GRU', X, y, session_lengths, nr_sessions=NR_SESSIONS, 
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)
    average_LSTM = session_cross_validation('LSTM', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)
    average_FFN = session_cross_validation('FFN', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)
    average_CNN = session_cross_validation('CNN_1D', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)

    print('\n')
    print('Cross-validated GRU:', average_GRU)
    print('Cross-validated LSTM:', average_LSTM)
    print('Cross-validated FFN:', average_FFN)
    print('Cross-validated CNN_1D:', average_CNN)
    print('\n')
    '''

    '''
    # ----- Inverse cross-validation ------
    # Trained on one session, tested on three
    average_GRU = inverse_session_cross_validation('GRU', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG, 
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)
    average_LSTM = inverse_session_cross_validation('LSTM', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)
    average_FFN = inverse_session_cross_validation('FFN', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)
    average_CNN = inverse_session_cross_validation('CNN_1D', X, y, session_lengths, nr_sessions=NR_SESSIONS,
                                                                        log_to_csv=LOG,
                                                                        batch_size=BATCH_SIZE, 
                                                                        epochs=EPOCHS)

    print('\n')
    print('Cross-validated one-session-train GRU:', average_GRU)
    print('Cross-validated one-session-train LSTM:', average_LSTM)
    print('Cross-validated one-session-train FFN:', average_FFN)
    print('Cross-validated one-session-train CNN_1D:', average_CNN)
    print('\n')
    '''

    # ----- PLOTTING ------

    #plot_comp_spread_single(X, y, session_lengths, NR_SESSIONS, epochs=30)
    #plot_comp_accuracy_single(X_soft, y_soft, session_lengths_soft, NR_SESSIONS, epochs=30)
    #plot_comp_val_SoftHard(X_soft, y_soft, X_hard, y_hard, session_lengths_soft, session_lengths_hard, NR_SESSIONS, epochs=30)
    #plot_comp_SoftHard_3(X_soft, y_soft, X_hard, y_hard, session_lengths_soft, session_lengths_hard, NR_SESSIONS, epochs=30)

    
