import numpy as np 
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from scipy.fft import fft, fftfreq
import pywt
from scipy.signal import wavelets
#import pyyawt

import Handle_emg_data as Handler

SAMPLE_RATE = 200

# Loads the data from the csv files into a storing system in an CSV_handler object
def load_user_emg_data():

    # CSV data from subject 1
    file1_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    file2_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1830/myoLeftEmg.csv"
    file3_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1845/myoLeftEmg.csv"
    file4_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1855/myoLeftEmg.csv"
    subject1_left_files = [file1_subject1_left, file2_subject1_left, file3_subject1_left, file4_subject1_left]
    file1_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoRightEmg.csv"
    file2_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1830/myoRightEmg.csv"
    file3_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1845/myoRightEmg.csv"
    file4_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1855/myoRightEmg.csv"
    subject1_right_files = [file1_subject1_rigth, file2_subject1_rigth, file3_subject1_rigth, file4_subject1_rigth]

    # CSV data from subject 2
    file1_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2010/myoLeftEmg.csv"
    file2_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2025/myoLeftEmg.csv"
    file3_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2035/myoLeftEmg.csv"
    file4_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2045/myoLeftEmg.csv"
    subject2_left_files = [file1_subject2_left, file2_subject2_left, file3_subject2_left, file4_subject2_left]
    file1_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2010/myoRightEmg.csv"
    file2_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2025/myoRightEmg.csv"
    file3_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2035/myoRightEmg.csv"
    file4_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2045/myoRightEmg.csv"
    subject2_right_files = [file1_subject2_rigth, file2_subject2_rigth, file3_subject2_rigth, file4_subject2_rigth]

    # CSV data from subject 3
    file1_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1700/myoLeftEmg.csv"
    file2_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1715/myoLeftEmg.csv"
    file3_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1725/myoLeftEmg.csv"
    file4_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1735/myoLeftEmg.csv"
    subject3_left_files = [file1_subject3_left, file2_subject3_left, file3_subject3_left, file4_subject3_left]
    file1_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1700/myoRightEmg.csv"
    file2_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1715/myoRightEmg.csv"
    file3_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1725/myoRightEmg.csv"
    file4_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1735/myoRightEmg.csv"
    subject3_right_files = [file1_subject3_rigth, file2_subject3_rigth, file3_subject3_rigth, file4_subject3_rigth]

    # CSV data from subject 4
    file1_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1900/myoLeftEmg.csv"
    file2_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1915/myoLeftEmg.csv"
    file3_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1925/myoLeftEmg.csv"
    file4_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1935/myoLeftEmg.csv"
    subject4_left_files = [file1_subject4_left, file2_subject4_left, file3_subject4_left, file4_subject4_left]
    file1_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1900/myoRightEmg.csv"
    file2_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1915/myoRightEmg.csv"
    file3_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1925/myoRightEmg.csv"
    file4_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1935/myoRightEmg.csv"
    subject4_right_files = [file1_subject4_rigth, file2_subject4_rigth, file3_subject4_rigth, file4_subject4_rigth]


    # CSV data from subject 5
    file1_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2030/myoLeftEmg.csv"
    file2_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2040/myoLeftEmg.csv"
    file3_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2050/myoLeftEmg.csv"
    file4_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2100/myoLeftEmg.csv"
    subject5_left_files = [file1_subject5_left, file2_subject5_left, file3_subject5_left, file4_subject5_left]
    file1_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2030/myoRightEmg.csv"
    file2_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2040/myoRightEmg.csv"
    file3_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2050/myoRightEmg.csv"
    file4_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2100/myoRightEmg.csv"
    subject5_right_files = [file1_subject5_rigth, file2_subject5_rigth, file3_subject5_rigth, file4_subject5_rigth]

    left_list = [subject1_left_files, subject2_left_files, subject3_left_files, subject4_left_files, subject5_left_files]
    right_list = [subject1_right_files, subject2_right_files, subject3_right_files, subject4_right_files, subject5_right_files]

    csv_handler = Handler.CSV_handler

    subject1_data_container = Handler.Data_container(1, 'HaluskaMarek')
    subject2_data_container = Handler.Data_container(1, 'HaluskaMaros')
    subject3_data_container = Handler.Data_container(1, 'HaluskovaBeata')
    subject4_data_container = Handler.Data_container(1, 'KelisekDavid')
    subject5_data_container = Handler.Data_container(1, 'KelisekRichard')
    subject_data_container_list = [subject1_data_container, subject2_data_container, subject3_data_container, 
                                    subject4_data_container, subject5_data_container]
    

    for subject_nr in range(5):
        # left variant proccessed here
        for round in range(4):
            for emg_nr in range(8):
                csv_handler.store_df(left_list[subject_nr][round], emg_nr+1, 'left', subject_data_container_list[subject_nr])
        # right variant proccessed here
        for round in range(4):
            for emg_nr in range(8):
                csv_handler.store_df(left_list[subject_nr][round], emg_nr+1, 'right', subject_data_container_list[subject_nr])

    return csv_handler.data_container_dict

# Takes in a df and outputs np arrays for x and y values
def get_xory_from_df(x_or_y, df:DataFrame):
    swither = {
        'x': df.iloc[:,0].to_numpy(),
        'y': df.iloc[:,1].to_numpy()
    }
    return swither.get(x_or_y, 0)

# Normalizes a ndarray of a signal to the scale of int16(32767)
def normalize_wave(y_values):
    y = np.int16((y_values / y_values.max()) * 32767)
    return y

# Takes the FFT of a DataFrame object
def fft_of_df(df:DataFrame):
    y_values = get_xory_from_df('y', df)
    N = y_values.size
    norm = normalize_wave(y_values)
    N_trans = fftfreq(N, 1 / SAMPLE_RATE)
    y_f = fft(norm)
    return N_trans, y_f, duration

# Removes noise with db4 wavelet function 
def wavelet_db4_denoising(df:DataFrame):
    y_values = get_xory_from_df('y', df)
    #y_values = normalize_wave(y_values)
    wavelet = pywt.Wavelet('db4')
    cA, cD = pywt.dwt(y_values, wavelet)
    N_trans =  np.array(range(int(np.floor((y_values.size + wavelet.dec_len - 1) / 2))))
    return N_trans, cA, cD 

# Filters signal accordning to Stein's Unbiased Risk Estimate(SURE)
def sure_threshold_filter(cA, cD):
    cA_filt = pyyawt.theselect(cA, 'rigrsure')
    cD_filt = cD 
    return cA_filt, cD_filt

# soft filtering of wavelet trans with the 40% lowest removed
def soft_threshold_filter(cA, cD):
    cA_filt = pywt.threshold(cA, 0.4 * cA.max())
    cD_filt = cD 
    return cA_filt, cD_filt

# Inverse dwt for brining denoise signal back to the time domainfi
def inverse_wavelet(df, cA_filt, cD_filt):
    wavelet = pywt.Wavelet('db4')
    y_new_values = pywt.idwt(cA_filt, cD_filt, wavelet)
    new_len = len(y_new_values)
    old_len = len(get_xory_from_df('y', df))
    if new_len > old_len:
        while new_len > old_len:
            y_new_values = y_new_values[:-1]
            new_len = len(y_new_values)
            old_len = len(get_xory_from_df('y', df))
    return y_new_values

# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
def plot_arrays(N, y):
    plt.plot(N, np.abs(y))
    plt.show()

