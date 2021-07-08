# Analysis of Keystroke EMG data for identification

#### EMG data handling and Neural Network analysis
Scripts to handle CSV files composed by 2 * 8 EMG sensors(left & right) devided into sessions per subject. The raw data is organised in a CSV_handler object with Handle_emg_data.py. Processing of data can take the further form of: 
* Preprocessing with Signal_prep.py - FFT, MFCC, Wavelet db4
* Storage for Neural Network analysis with NN_handler(Handle_emg_data.py) - combined EMG DataFrame, combined MFCCs DataFrame
* Neural Network analysis in Neural_Network_Analysis.py - LSTM NN, etc.

#### Technologies used
* Common libs: Numpy, Pandas, Pathlib, Sklearn, Scipy, Matplotlib, Tensorflow, Keras
* Community libs: Python_speech_features, Pywt

#### Challanges in the module
* The CSV handlig is for the moment hard-coded to fit the current project due to a very specific file structure and respective naming convention.
* Preprocessing is still limited in Signal_prep.py
* Neural_Network_Analysis.py lacks a more general way to access multiple types of networks

#### Credits for insporational code

* Kapre - Keunwoochoi
* Audio-Classification: seth814
* DeepLearningForAudioWithPyhton - musikalkemist

## Table of Contents

| File and classes                                                            | Description and help functions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Handle_emg_data.py:     - Data_container     - CSV_handler     - NN_handler | Handles, manipulates, and stores data for analysis.       - Data_container is a class that describes the data for each subject in the experiment.        Use __init__.      - CSV_handler takes data from CSV files and places it in Data_container for each subject.       Use load_data() to load csv data into data containers and add the containers to the        CSV_handler's 'data_container_dict', indexed by subject number. Use get_data() to retrieve        specific data.      - NN_handler prepares data for further analysis in Neural Networks. This class has storage        for this data and/or can save it to a json file. |
| Signal_prep.py                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Present_data.py                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Neural_Network_Analysis.py                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

