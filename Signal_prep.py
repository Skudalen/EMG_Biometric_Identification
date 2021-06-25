import numpy as np 
import matplotlib.pyplot as plt
import pandas
from pandas.core.frame import DataFrame
from scipy.fft import fft, fftfreq
import pywt
#from scipy.signal import wavelets
#import pyyawt

import Handle_emg_data as Handler

SAMPLE_RATE = 200


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
    return N_trans, y_f

# Removes noise with db4 wavelet function 
def wavelet_db4_denoising(df:DataFrame):
    y_values = get_xory_from_df('y', df)
    #y_values = normalize_wave(y_values)
    wavelet = pywt.Wavelet('db4')
    cA, cD = pywt.dwt(y_values, wavelet)
    N_trans =  np.array(range(int(np.floor((y_values.size + wavelet.dec_len - 1) / 2))))
    return N_trans, cA, cD 

# Filters signal accordning to Stein's Unbiased Risk Estimate(SURE)
'''
def sure_threshold_filter(cA, cD):
    cA_filt = pyyawt.theselect(cA, 'rigrsure')
    cD_filt = cD 
    return cA_filt, cD_filt
'''

# soft filtering of wavelet trans with the 40% lowest removed
def soft_threshold_filter(cA, cD, threshold):
    cA_filt = pywt.threshold(cA, threshold * cA.max())
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

# Takes in handler and detailes to denoise. Returns arrays and df
def denoice_dataset(handler:Handler.CSV_handler, subject_nr, which_arm, round, emg_nr, threshold):
    df = handler.get_df_from_data_dict(subject_nr, which_arm, round, emg_nr)

    N = get_xory_from_df('x', df)
    N_trans, cA, cD = wavelet_db4_denoising(df)
    cA_filt, cD_filt = soft_threshold_filter(cA, cD, threshold)
    y_values = inverse_wavelet(df, cA_filt, cD_filt)

    df_new = Handler.make_df_from_xandy(N, y_values, emg_nr)
    return df_new



# MOVE TO Present_data.py
# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
def plot_arrays(N, y):
    plt.plot(N, np.abs(y))
    plt.show()

