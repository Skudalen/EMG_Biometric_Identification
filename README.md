# Analysis of Keystroke EMG data for identification

###### EMG data handling and Neural Network analysis
    Scripts to handle CSV files composed by 2 * 8 EMG sensors(left & right) devided into sessions per subject. The raw data is organised in a CSV_handler object with Handle_emg_data.py. Processing of data can take the further form of: 
    * Preprocessing with Signal_prep.py - FFT, MFCC, Wavelet db4
    * Storage for Neural Network analysis with DL_handler(Handle_emg_data.py) - combined EMG DataFrame, combined MFCCs DataFrame
    * Neural Network analysis in Neural_Network_Analysis.py - LSTM NN, etc.

 ###### Technologies used
    * Common libs: Numpy, Pandas, Pathlib, Sklearn, Scipy, Matplotlib, Tensorflow, Keras
    * Indi libs: Python_speech_features, Pywt

###### Challanges in the module
    * The CSV handlig is for the moment hard-coded to fit the current project due to a very specific file structure and respective naming convention.
    * Preprocessing is still limited in Signal_prep.py
    * Neural_Network_Analysis.py lacks a more general way to access multiple types of networks

