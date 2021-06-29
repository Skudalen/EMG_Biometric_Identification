from logging import error

from matplotlib.cbook import get_sample_data
from Handle_emg_data import *
from Signal_prep import *
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker

# Global variables for MFCC
mfcc_stepsize = 0.5
mfcc_windowsize = 2


# PLOT FUNCTIONS --------------------------------------------------------------: 

# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
def plot_array(N, y):
    plt.plot(N, np.abs(y))
    plt.show()

def plot_compare_two_df(df_old, old_name, df_new, new_name):
    x = get_xory_from_df('x', df_old)
    y1 = get_xory_from_df('y', df_old)
    y2 = get_xory_from_df('y', df_new)

    figure, axis = plt.subplots(1, 2)
    axis[0].plot(x, y1)
    axis[0].set_title(old_name)
    axis[1].plot(x, y2)
    axis[1].set_title(new_name)
    plt.show()

def plot_mfcc(mfcc_data, data_label:str):
    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
    
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * mfcc_stepsize))
    ax.xaxis.set_major_formatter(ticks_x)

    ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC: ' + data_label)
    ax.set_ylabel('Cepstral Coefficients')
    ax.set_xlabel('Time(s)')
    plt.show() 

def plot_3_mfcc(mfcc_data1, data_label1:str, mfcc_data2, data_label2:str, mfcc_data3, data_label3:str):
    
    fig, axes = plt.subplots(nrows=3)
    plt.subplots_adjust(hspace=1.4, wspace=0.4)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * mfcc_stepsize))

    data_list = [mfcc_data1, mfcc_data2, mfcc_data3]
    label_list = [data_label1, data_label2, data_label3]

    for ax, data, label in zip(axes, data_list, label_list):
        mfcc_data= np.swapaxes(data, 0 ,1)
        ax.xaxis.set_major_formatter(ticks_x)
   
        ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
        ax.set_title('MFCC: ' + str(label))
        ax.set_ylabel('Coefficients')
        ax.set_xlabel('Time(s)')

    plt.show() 

def plot_all_emg_mfcc(data_list:list, label_list:list):
    fig, axes = plt.subplots(nrows=4, ncols=2)
    plt.subplots_adjust(hspace=1.4, wspace=0.4)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * mfcc_stepsize))
    plt.autoscale()

    d_list = np.array([ [data_list[0], data_list[4]],
                        [data_list[1], data_list[5]],
                        [data_list[2], data_list[6]],
                        [data_list[3], data_list[7]]
                      ])
    l_list = np.array([ [label_list[0], label_list[4]],
                        [label_list[1], label_list[5]],
                        [label_list[2], label_list[6]],
                        [label_list[3], label_list[7]]
                      ])

    for col in [0, 1]:
        for ax, data, label in zip(axes[:,col], d_list[:,col], l_list[:,col]):
            mfcc_data= np.swapaxes(data, 0 ,1)
            ax.xaxis.set_major_formatter(ticks_x)
    
            ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
            ax.set_title('MFCC: ' + str(label))
            ax.set_ylabel('Coefficients')
            ax.set_xlabel('Time(s)')

    plt.show() 

# DATA FUNCTIONS: --------------------------------------------------------------: 

# The CSV_handler takes in data_type, but only for visuals. 
# E.g. handler = CSV_handler('soft')

# Loads in data to a CSV_handler. Choose data_type: hard, hardPP, soft og softPP as str. 
# Returns None. 
def load_data(csv_handler:CSV_handler, data_type):
    if data_type == 'hard': 
        csv_handler.load_hard_original_emg_data()
    elif data_type == 'hardPP':
        csv_handler.load_hard_PP_emg_data()
    elif data_type == 'soft':
        csv_handler.load_soft_original_emg_data()
    elif data_type == 'softPP':
        csv_handler.load_soft_PP_emg_data()
    else:
        raise Exception('Wrong input')

# Retrieved data. Send in loaded csv_handler and data detailes you want. 
# Returns DataFrame and samplerate
def get_data(csv_handler:CSV_handler, subject_nr, which_arm, session, emg_nr):
    data_frame = csv_handler.get_df_from_data_dict(subject_nr, which_arm, session, emg_nr)
    samplerate = get_samplerate(data_frame)
    return data_frame, samplerate
    
# Takes in handler and detailes to denoise. 
# Returns arrays and df
def denoice_dataset(handler:Handler.CSV_handler, subject_nr, which_arm, round, emg_nr):
    df = handler.get_df_from_data_dict(subject_nr, which_arm, round, emg_nr)

    N = get_xory_from_df('x', df)
    N_trans, cA, cD = wavelet_db4(df)
    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    y_values = inverse_wavelet(df, cA_filt, cD_filt)

    df_new = Handler.make_df_from_xandy(N, y_values, emg_nr)
    return df_new

# Slightly modified mfcc with inputs like below.
# Returns N (x_values from original df) and mfcc_y_values 
def mfcc_custom(df:DataFrame, samplesize, windowsize, stepsize):
    N = get_xory_from_df('x', df)
    y = get_xory_from_df('y', df)
    return N, base.mfcc(y, samplesize, windowsize, stepsize)


# CASE FUNTIONS ----------------------------------------------------------------: 

# Takes in a df and compares the FFT and the wavelet denoising of the FFT
# Returns None. Plots the two
def compare_with_wavelet_filter(data_frame):
    N_trans, cA, cD = wavelet_db4(data_frame)
    data_frame_freq = make_df_from_xandy(N_trans, cA, 1)

    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    data_frame_freq_filt = make_df_from_xandy(N_trans, cD_filt, 1)

    plot_compare_two_df(data_frame_freq, 'Original data', data_frame_freq_filt, 'Analyzed data')

# Loads three preset emg_1 datasets(subj1:session1, subj1:session2, subj2:session1), calculates mfcc for each and plots them.
# Input: CSV_handler
# Output: None --> Plot
def mfcc_3_plots_1_1_2(csv_handler:CSV_handler):
    df1, samplerate1 = get_data(csv_handler, 1, 'left', 1, 1)
    df2, samplerate2 = get_data(csv_handler, 1, 'left', 2, 1)
    df3, samplerate3 = get_data(csv_handler, 2, 'left', 1, 1)
    #print(df1.head, samplerate1)
    #print(df2.head, samplerate2)
    #print(df3.head, samplerate3)
    N1, mfcc_feat1 = mfcc_custom(df1, samplerate1, mfcc_windowsize, mfcc_stepsize)
    N2, mfcc_feat2 = mfcc_custom(df2, samplerate2, mfcc_windowsize, mfcc_stepsize)
    N3, mfcc_feat3 = mfcc_custom(df3, samplerate3, mfcc_windowsize, mfcc_stepsize)
    label_1 = 'Subject 1, session 1, left arm, emg nr. 1'
    label_2 = 'Subject 1, session 2, left arm, emg nr. 1'
    label_3 = 'Subject 2, session 1, left arm, emg nr. 1'

    plot_3_mfcc(mfcc_feat1, label_1, mfcc_feat2, label_2, mfcc_feat3, label_3)

# Loads three preset emg_1 datasets(subj3:session1, subj3:session2, subj4:session1), calculates mfcc for each and plots them.
# Input: CSV_handler
# Output: None --> Plot
def mfcc_3_plots_3_3_4(csv_handler:CSV_handler):
    df1, samplerate1 = get_data(csv_handler, 3, 'left', 1, 1)
    df2, samplerate2 = get_data(csv_handler, 3, 'left', 2, 1)
    df3, samplerate3 = get_data(csv_handler, 4, 'left', 1, 1)
    #print(df1.head, samplerate1)
    #print(df2.head, samplerate2)
    #print(df3.head, samplerate3)
    N1, mfcc_feat1 = mfcc_custom(df1, samplerate1, mfcc_windowsize, mfcc_stepsize)
    N2, mfcc_feat2 = mfcc_custom(df2, samplerate2, mfcc_windowsize, mfcc_stepsize)
    N3, mfcc_feat3 = mfcc_custom(df3, samplerate3, mfcc_windowsize, mfcc_stepsize)
    label_1 = 'Subject 3, session 1, left arm, emg nr. 1'
    label_2 = 'Subject 3, session 2, left arm, emg nr. 1'
    label_3 = 'Subject 4, session 1, left arm, emg nr. 1'

    plot_3_mfcc(mfcc_feat1, label_1, mfcc_feat2, label_2, mfcc_feat3, label_3)

def mfcc_all_emg_plots(csv_handler:CSV_handler):
    df1, samplerate1 = get_data(csv_handler, 1, 'left', 1, 1)
    df2, samplerate2 = get_data(csv_handler, 1, 'left', 1, 2)
    df3, samplerate3 = get_data(csv_handler, 1, 'left', 1, 3)
    df4, samplerate4 = get_data(csv_handler, 1, 'left', 1, 4)
    df5, samplerate5 = get_data(csv_handler, 1, 'left', 1, 5)
    df6, samplerate6 = get_data(csv_handler, 1, 'left', 1, 6)
    df7, samplerate7 = get_data(csv_handler, 1, 'left', 1, 7)
    df8, samplerate8 = get_data(csv_handler, 1, 'left', 1, 8)
    N1, mfcc_feat1 = mfcc_custom(df1, samplerate1, mfcc_windowsize, mfcc_stepsize)
    N2, mfcc_feat2 = mfcc_custom(df2, samplerate2, mfcc_windowsize, mfcc_stepsize)
    N3, mfcc_feat3 = mfcc_custom(df3, samplerate3, mfcc_windowsize, mfcc_stepsize)
    N4, mfcc_feat4 = mfcc_custom(df4, samplerate4, mfcc_windowsize, mfcc_stepsize)
    N5, mfcc_feat5 = mfcc_custom(df5, samplerate5, mfcc_windowsize, mfcc_stepsize)
    N6, mfcc_feat6 = mfcc_custom(df6, samplerate6, mfcc_windowsize, mfcc_stepsize)
    N7, mfcc_feat7 = mfcc_custom(df7, samplerate7, mfcc_windowsize, mfcc_stepsize)
    N8, mfcc_feat8 = mfcc_custom(df8, samplerate8, mfcc_windowsize, mfcc_stepsize)
    feat_list = [mfcc_feat1, mfcc_feat2, mfcc_feat3, mfcc_feat4, mfcc_feat5, mfcc_feat6, mfcc_feat7, mfcc_feat8]
    label_1 = 'Subject 1, session 1, left arm, emg nr. 1'
    label_2 = 'Subject 1, session 1, left arm, emg nr. 2'
    label_3 = 'Subject 1, session 1, left arm, emg nr. 3'
    label_4 = 'Subject 1, session 1, left arm, emg nr. 4'
    label_5 = 'Subject 1, session 1, left arm, emg nr. 5'
    label_6 = 'Subject 1, session 1, left arm, emg nr. 6'
    label_7 = 'Subject 1, session 1, left arm, emg nr. 7'
    label_8 = 'Subject 1, session 1, left arm, emg nr. 8'
    label_list = [label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8]

    plot_all_emg_mfcc(feat_list, label_list)

# MAIN: ------------------------------------------------------------------------: 

def main():

    csv_handler = CSV_handler()
    load_data(csv_handler, 'soft')
    #mfcc_3_plots_1_1_2(csv_handler)
    #mfcc_3_plots_3_3_4(csv_handler)
    mfcc_all_emg_plots(csv_handler)
    

main()