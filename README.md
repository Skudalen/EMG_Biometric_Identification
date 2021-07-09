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

* Kapre: Keunwoochoi
* Audio-Classification: seth814
* DeepLearningForAudioWithPyhton: musikalkemist

## Table of Contents

| File and classes | Description and help functions |
|---|---|
| Handle_emg_data.py:<br><br>    * Data_container<br>    * CSV_handler<br>    * NN_handler | Handles, manipulates, and stores data for analysis. <br><br>    * Data_container is a class that describes the data for each subject in the experiment.<br>    * CSV_handler takes data from CSV files and places it in Data_container for each subject.<br>      Use load_data() to load csv data into data containers and add the containers to the <br>      CSV_handler's 'data_container_dict', indexed by subject number. Use get_data() to retrieve <br>      specific data. <br>    * NN_handler prepares data for further analysis in Neural Networks. This class has storage <br>      for this data and/or can save it to a json file. |
| Signal_prep.py | Does mapping to data and contains various functions. Among others, this contains wavelet, <br>MFCC, cepstrum and normalization.  |
| Present_data.py  | Contains plot and case functions. Case functions combines many elements from the code and <br>presents some results described. |
| Neural_Network_Analysis.py | Contains functions to load, build and execute analysis with Neural Networks. Main functions are <br>load_data_from_json(), build_model(), and main() |


## How to use it 

1. Clone the repo
2. Place the data files in the working directory 
3. (For now) Add the session filenames in the desired load_data() function 
4. Assuming NN analysis:
    1. Create a `CSV_handler` object 
    2. Load data with `load_data(CSV_handler, <datatype>)`
    3. Create `NN_handler` object with `CSV_handler` as input
    4. Load MFCC data into the `NN_handler` with `store_mfcc_samples()`
    5. Run `save_json_mfcc()` to save samples in json
    6. Run `Neural_Network_Analysis.py` with desired config

