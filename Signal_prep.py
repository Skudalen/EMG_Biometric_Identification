import numpy as np 
import matplotlib.pyplot as plt
import pandas
from pandas.core.frame import DataFrame
from scipy.fft import fft, fftfreq
import pywt
import pyhton_speech_features as psf
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
def wavelet_db4(df:DataFrame):
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

# soft filtering of wavelet trans with the a 1/2 std filter 
def soft_threshold_filter(cA, cD):
    cA_filt = pywt.threshold(cA, np.std(cA)/2)
    cD_filt = pywt.threshold(cD, np.std(cD)/2)
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

def cepstrum(df:DataFrame):
    N = get_xory_from_df('x', df)
    y = get_xory_from_df('y', df)
    
    
    return None

'''
def mfcc(df:DataFrame):
    N = get_xory_from_df('x', df)
    y = get_xory_from_df('y', df)
    spf.mfcc(y, )
'''
