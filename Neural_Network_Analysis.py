import json
from python_speech_features.python_speech_features.base import mfcc
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Path to json file that stores MFCCs and subject labels for each processed sample
DATA_PATH_MFCC = str(Path.cwd()) + "/mfcc_data.json"

# Loads data from the json file and reshapes X_data(samples, 1, 208) and y_data(samples, 1)
# Input: JSON path
# Ouput: X(mfcc data), y(labels)
def load_data_from_json(data_path): 

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arraysls
    X = np.array(data['mfcc'])
    X = X.reshape(X.shape[0], 1, X.shape[1])
    #print(X.shape)
    
    y = np.array(data["labels"])
    y = y.reshape(y.shape[0], 1)
    #print(y.shape)
    

    print("Data succesfully loaded!")

    return X, y

# Plots the training history with two subplots. First training and test accuracy, and then 
# loss with respect to epochs
# Input: History(from model.fit(...))
# Ouput: None -> plot
def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

# Takes in data and labels, and splits it into train, validation and test sets
# Input: Data, labels, whether to shuffle, % validatiion, % test
# Ouput: X_train, X_validation, X_test, y_train, y_validation, y_test
def prepare_datasets_percentsplit(X, y, shuffle_vars:bool, validation_size=0.2, test_size=0.25,):

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle_vars)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, shuffle=shuffle_vars)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# Creates a RNN_LSTM neural network model
# Input: input shape, classes of classification
# Ouput: model:Keras.model
def RNN_LSTM(input_shape, nr_classes=5):
    """Generates RNN-LSTM model
    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(nr_classes, activation='softmax'))

    return model

# Trains the model 
# Input: Keras.model, batch_size, nr epochs, training, and validation data
# Ouput: History
def train(model, batch_size, epochs, X_train, X_validation, y_train, y_validation):

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, 
                        y_train, 
                        validation_data=(X_validation, y_validation), 
                        batch_size=batch_size, 
                        epochs=epochs)
    return history


if __name__ == "__main__":

    # Load data
    X, y = load_data_from_json(DATA_PATH_MFCC)

    # Get prepared data: train, validation, and test
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets_percentsplit(X, y,
                                                                                                validation_size=0.2, 
                                                                                                test_size=0.25,  
                                                                                                shuffle_vars=True)
    print(X_train.shape)

    # Make model
    model = RNN_LSTM(input_shape=(1, 208))
    model.summary()
    
    # Train network
    history = train(model, X_train, X_validation, y_train, y_validation, batch_size=64, epochs=30)
    
    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    


