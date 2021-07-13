import json
from python_speech_features.python_speech_features.base import mfcc
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.regularizers import l2
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# Path to json file that stores MFCCs and subject labels for each processed sample
DATA_PATH_MFCC = str(Path.cwd()) + "/mfcc_data.json"

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

# Takes in data and labels, and splits it into train, validation and test sets by percentage
# Input: Data, labels, whether to shuffle, % validatiion, % test
# Ouput: X_train, X_validation, X_test, y_train, y_validation, y_test
def prepare_datasets_percentsplit(X, y, shuffle_vars, validation_size=0.2, test_size=0.25,):

    # Create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle_vars)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, shuffle=shuffle_vars)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# Takes in data, labels, and session_lengths and splits it into train and test sets by session_index
# Input: Data, labels, session_lengths, test_session_index
# Ouput: X_train, X_test, y_train, y_test
def prepare_datasets_sessions(X, y, session_lengths, test_session_index=4, nr_subjects=5):

    session_lengths = session_lengths.tolist()

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

# Trains the model 
# Input: Keras.model, batch_size, nr epochs, training, and validation data
# Ouput: History
def train(model, X_train, y_train, verbose, batch_size=64, epochs=30, X_validation=None, y_validation=None):

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if X_validation != None:
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

# Retrieves data sets for each session as test set and evalutes 
# the average of networks trained om them
# Input: raw data, session_lengths list, total nr of sessions, batch_size, and nr of epochs 
# Ouput: tuple(cross validation average, list(result for each dataset(len=nr_sessions)))
def session_cross_validation(model_name:str, X, y, session_lengths, nr_sessions, batch_size=64, epochs=30):
    session_training_results = []
    for i in range(nr_sessions):

        # Model:
        if model_name == 'LSTM':
            model = LSTM(input_shape=(1, 208))
        elif model_name == 'GRU':
            model = GRU(input_shape=(1, 208))
        elif model_name == 'CNN':
            continue
            model = CNN(input_shape=(1, 208))
        elif model_name == 'FNN':
            model = FFN(input_shape=(1, 208))
        else:
            raise Exception('Model not found')

        model.summary()

        X_train_session, X_test_session, y_train_session, y_test_session = prepare_datasets_sessions(X, y, session_lengths, i)
        train(model, X_train_session, y_train_session, verbose=1, batch_size=batch_size, epochs=epochs)
        test_loss, test_acc = model.evaluate(X_test_session, y_test_session, verbose=2)
        session_training_results.append(test_acc)
        del model
        K.clear_session()
        #print('Session', i, 'as test data gives accuracy:', test_acc)

    average_result = statistics.mean((session_training_results))

    return average_result, session_training_results


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
    model.add(keras.layers.Dense(nr_classes, activation='softmax', name='Dense_relu_output'))

    return model

# Creates a keras.model with focus on GRU layers
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
    model.add(keras.layers.Dense(nr_classes, activation='softmax', name='Dense_relu_output'))

    return model



if __name__ == "__main__":

    # ----- Load data ------
        # X.shape = (2806, 1, 208)
        # y.shape = (2806, nr_subjects)
        # session_lengths.shape = (nr_subjects, nr_sessions)
    X, y, session_lengths = load_data_from_json(DATA_PATH_MFCC, nr_classes=5)

    # Parameters:
    NR_SUBJECTS = 5
    NR_SESSIONS = 4
    BATCH_SIZE = 64
    EPOCHS = 30

    TEST_SESSION_NR = 4
    VERBOSE = 0 
    
    # ----- Get prepared data: train, validation, and test ------
        # X_train.shape = (2806-X_test, 1, 208)
        # X_test.shape = (X_test(from session nr. ?), 1, 208)
        # y_train.shape = (2806-y_test, nr_subjects)
        # y_test.shape = (y_test(from session nr. ?), nr_subjects)

    X_train, X_test, y_train, y_test = prepare_datasets_sessions(X, y, session_lengths, TEST_SESSION_NR)
    

    #'''
    # ----- Make model ------
    #model_GRU = GRU(input_shape=(1, 208)) # (timestep, 13*16 MFCC coefficients)
    #model_LSTM = LSTM(input_shape=(1, 208)) # (timestep, 13*16 MFCC coefficients)
    
    #model_GRU.summary()
    #model_LSTM.summary()
    
    # ----- Train network ------
    #history_GRU = train(model_GRU, X_train, y_train, verbose=VERBOSE, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #history_LSTM = train(model_LSTM, X_train, y_train, verbose=VERBOSE, batch_size=BATCH_SIZE, epochs=EPOCHS)

    #average_GRU = session_cross_validation('GRU', X, y, session_lengths, NR_SESSIONS, BATCH_SIZE, EPOCHS)
    #verage_LSTM = session_cross_validation('LSTM', X, y, session_lengths, NR_SESSIONS, BATCH_SIZE, EPOCHS)
    average_FFN = session_cross_validation('FNN', X, y, session_lengths, NR_SESSIONS, BATCH_SIZE, EPOCHS)
    print('\n')
    #print('Crossvalidated GRU:', average_GRU)
    #print('Crossvalidated LSTM:', average_LSTM)
    print('Crossvalidated FFN:', average_FFN)
    print('\n')
    
    # ----- Plot train accuracy/error -----
    #plot_train_history(history)

    # ----- Evaluate model on test set ------
    #test_loss, test_acc = model_GRU.evaluate(X_test, y_test, verbose=VERBOSE)
    #print('\nTest accuracy GRU:', test_acc, '\n')
    #test_loss, test_acc = model_LSTM.evaluate(X_test, y_test, verbose=VERBOSE)
    #print('\nTest accuracy LSTM:', test_acc, '\n')
    #'''
    


