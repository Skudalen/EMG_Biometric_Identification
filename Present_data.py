from logging import error

from matplotlib.cbook import get_sample_data
from Handle_emg_data import *
from Signal_prep import *
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker

# Global variables for MFCC
MFCC_STEPSIZE = 0.5     # Seconds
MFCC_WINDOWSIZE = 2     # Seconds
NR_COEFFICIENTS = 13    # Number of coefficients
NR_MEL_BINS = 40     # Number of mel-filter-bins 


# PLOT FUNCTIONS --------------------------------------------------------------: 

# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
# Input: X-values and Y-values
# Output: None --> Plot
def plot_array(N, y):
    plt.plot(N, np.abs(y))
    plt.show()

# Plots two subplots with two dataframes in order to compare them
# Input: Old dataframe, old title, new dataframe, new title
# Output: None --> Plot
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

# Plots one set of MFCC data
# Input: 2d array of MFCC data(frame, coefficients), data_label for description
# Output: None -> Plot
def plot_mfcc(mfcc_data, data_label:str):
    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
    
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * MFCC_STEPSIZE))
    ax.xaxis.set_major_formatter(ticks_x)

    ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC: ' + data_label)
    ax.set_ylabel('Cepstral Coefficients')
    ax.set_xlabel('Time(s)')
    plt.show() 

# Plots three sets of MFCC data
# Input: 3 x (2d array of MFCC data(frame, coefficients)), 3 x (data_label for description)
# Output: None -> Plot
def plot_3_mfcc(mfcc_data1, data_label1:str, mfcc_data2, data_label2:str, mfcc_data3, data_label3:str):
    
    fig, axes = plt.subplots(nrows=3)
    plt.subplots_adjust(hspace=1.4, wspace=0.4)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * MFCC_STEPSIZE))

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

# Plots eight subplots with all EMG data from Subject 1 and Session 1
# Input: list of 8 arrays of EMG data(datapoints), list of 8 data_labels for description
# Output: None -> Plot
def plot_all_emg_mfcc(data_list:list, label_list:list):
    fig, axes = plt.subplots(nrows=4, ncols=2)
    plt.subplots_adjust(hspace=1.4, wspace=0.4)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * MFCC_STEPSIZE))
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

def pretty(dict):
    for key, value in dict.items():
        print('Subject', key, 'samples:')
        print('\t\t Number av samples:', len(value)) 
        print('\t\t EX sample nr 1:') 
        print('\t\t\t Type:', type(value[0][0]), type(value[0][1]))
        print('\t\t\t Sample:', value[0][0], value[0][1])


# DATA FUNCTIONS: --------------------------------------------------------------: 

# The CSV_handler takes in nr of subjects and nr of sessions in the experiment
# E.g. handler = CSV_handler(nr_subjects=5, nr_sessions=4)
# Needs to load data: handler.load_data(<type>, <type_directory_name>)
    
# Denoices one set of EMG data 
# Input: CSV_handler and detailes for ID 
# Output: DataFrame(df)
def denoice_dataset(handler:CSV_handler, subject_nr, which_arm, round, emg_nr):
    df = handler.get_df_from_data_dict(subject_nr, which_arm, round, emg_nr)

    N = get_xory_from_df('x', df)
    N_trans, cA, cD = wavelet_db4(df)
    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    y_values = inverse_wavelet(df, cA_filt, cD_filt)

    df_new = make_df_from_xandy(N, y_values, emg_nr)
    return df_new

# Quick debug function for NN_handler dict
# Input: NN_hanlder dict, nr of samples per person
# Output: None -> prints if NaN
def test_for_NaN(dict, samples_per_person):
    for key, value in dict.items():
        for i in range(samples_per_person):
            df = value[i][0]
            #print(df)
            print(df.isnull())


# CASE FUNTIONS ----------------------------------------------------------------: 

# Takes in a df and compares the FFT and the wavelet denoising of the FFT
# Input: timestamp/EMG Dataframe
# Output: None --> Plot
def compare_with_wavelet_filter(data_frame):
    N_trans, cA, cD = wavelet_db4(data_frame)
    data_frame_freq = make_df_from_xandy(N_trans, cA, 1)

    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    data_frame_freq_filt = make_df_from_xandy(N_trans, cD_filt, 1)

    plot_compare_two_df(data_frame_freq, 'Original data', data_frame_freq_filt, 'Analyzed data')

# Loads three preset EMG nr 1 datasets(subj1:session1, subj1:session2, subj2:session1), calculates mfcc for each and plots them.
# Input: CSV_handler
# Output: None --> Plot
def mfcc_3_plots_1_1_2(csv_handler:CSV_handler):
    df1, samplerate1 = csv_handler.get_data( 1, 'left', 1, 1)
    df2, samplerate2 = csv_handler.get_data( 1, 'left', 2, 1)
    df3, samplerate3 = csv_handler.get_data( 2, 'left', 1, 1)
    #print(df1.head, samplerate1)
    #print(df2.head, samplerate2)
    #print(df3.head, samplerate3)
    N1, mfcc_feat1 = mfcc_custom(df1, samplerate1, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N2, mfcc_feat2 = mfcc_custom(df2, samplerate2, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N3, mfcc_feat3 = mfcc_custom(df3, samplerate3, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    label_1 = 'Subject 1, session 1, left arm, emg nr. 1'
    label_2 = 'Subject 1, session 2, left arm, emg nr. 1'
    label_3 = 'Subject 2, session 1, left arm, emg nr. 1'

    plot_3_mfcc(mfcc_feat1, label_1, mfcc_feat2, label_2, mfcc_feat3, label_3)

# Loads three preset EMG nr 1 datasets(subj3:session1, subj3:session2, subj4:session1), calculates mfcc for each and plots them.
# Input: CSV_handler
# Output: None --> Plot
def mfcc_3_plots_3_3_4(csv_handler:CSV_handler):
    df1, samplerate1 = csv_handler.get_data(3, 'left', 1, 1)
    df2, samplerate2 = csv_handler.get_data(3, 'left', 2, 1)
    df3, samplerate3 = csv_handler.get_data(4, 'left', 1, 1)
    #print(df1.head, samplerate1)
    #print(df2.head, samplerate2)
    #print(df3.head, samplerate3)
    N1, mfcc_feat1 = mfcc_custom(df1, samplerate1, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N2, mfcc_feat2 = mfcc_custom(df2, samplerate2, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N3, mfcc_feat3 = mfcc_custom(df3, samplerate3, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    label_1 = 'Subject 3, session 1, left arm, emg nr. 1'
    label_2 = 'Subject 3, session 2, left arm, emg nr. 1'
    label_3 = 'Subject 4, session 1, left arm, emg nr. 1'

    plot_3_mfcc(mfcc_feat1, label_1, mfcc_feat2, label_2, mfcc_feat3, label_3)

# Loads preset emg 1-8 datasets(subj1 and session1) and calculates mfcc for each and plots them.
# Input: CSV_handler
# Output: None --> Plot
def mfcc_all_emg_plots(csv_handler:CSV_handler):
    df1, samplerate1 = csv_handler.get_data( 1, 'left', 1, 1)
    df2, samplerate2 = csv_handler.get_data( 1, 'left', 1, 2)
    df3, samplerate3 = csv_handler.get_data( 1, 'left', 1, 3)
    df4, samplerate4 = csv_handler.get_data( 1, 'left', 1, 4)
    df5, samplerate5 = csv_handler.get_data( 1, 'left', 1, 5)
    df6, samplerate6 = csv_handler.get_data( 1, 'left', 1, 6)
    df7, samplerate7 = csv_handler.get_data( 1, 'left', 1, 7)
    df8, samplerate8 = csv_handler.get_data( 1, 'left', 1, 8)
    N1, mfcc_feat1 = mfcc_custom(df1, samplerate1, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N2, mfcc_feat2 = mfcc_custom(df2, samplerate2, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N3, mfcc_feat3 = mfcc_custom(df3, samplerate3, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N4, mfcc_feat4 = mfcc_custom(df4, samplerate4, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N5, mfcc_feat5 = mfcc_custom(df5, samplerate5, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N6, mfcc_feat6 = mfcc_custom(df6, samplerate6, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N7, mfcc_feat7 = mfcc_custom(df7, samplerate7, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
    N8, mfcc_feat8 = mfcc_custom(df8, samplerate8, MFCC_WINDOWSIZE, MFCC_STEPSIZE)
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

# Prints (and logs) max, min, mean, EMS and median of EMG data
# Input: CSV_handler
# Output: None --> Print
def log_emg_characteristics(csv_handler:CSV_handler):

    min_values = []
    max_values = []
    mean_list = []
    RMS_list = [] 
    median_list = []

    if csv_handler.data_type == 'soft':

        for subject_container in csv_handler.data_container_dict.values():
            min_values_sub = []
            max_values_sub = []
            mean_list_sub = []
            RMS_list_sub = []  
            median_list_sub = []
            for session_dict in subject_container.dict_list:
                for emg_list in session_dict.values():
                    for emg_df in emg_list:

                        df = emg_df.iloc[:,1]
                        min_values_sub.append(df.min())
                        max_values_sub.append(df.max())
                        mean_list_sub.append(df.abs().mean())
                        RMS_list_sub.append(np.sqrt(np.mean(np.square(df.to_numpy()))))
                        median_list_sub.append(df.abs().median())

                        min_values.append(df.min())
                        max_values.append(df.max())
                        mean_list.append(df.abs().mean())
                        RMS_list.append(np.sqrt(np.mean(np.square(df.to_numpy()))))
                        median_list.append(df.abs().median())
            
            subject_nr = subject_container.subject_nr
            #print('\n')
            print('Natural typing behavior, subject {}, minimum EMG value:'.format(subject_nr), min(min_values_sub))
            print('Natural typing behavior, subject {}, maximum EMG value:'.format(subject_nr), max(max_values_sub))
            print('Natural typing behavior, subject {}, mean EMG value:'.format(subject_nr), np.mean(mean_list_sub))
            print('Natural typing behavior, subject {}, RMS EMG value:'.format(subject_nr), np.sqrt(np.mean(np.square(RMS_list_sub))))
            print('Natural typing behavior, subject {}, median EMG value:'.format(subject_nr), np.median(median_list_sub))
            print('\n')
    
    elif csv_handler.data_type == 'hard':

        for subject_container in csv_handler.data_container_dict.values():
            min_values_sub = []
            max_values_sub = []
            mean_list_sub = []
            RMS_list_sub = []  
            median_list_sub = []
            for session_dict in subject_container.dict_list:
                for emg_list in session_dict.values():
                    for emg_df in emg_list:

                        df = emg_df.iloc[:,1]
                        min_values_sub.append(df.min())
                        max_values_sub.append(df.max())
                        mean_list_sub.append(df.abs().mean())
                        RMS_list_sub.append(np.sqrt(np.mean(np.square(df.to_numpy()))))
                        median_list_sub.append(df.abs().median())

                        min_values.append(df.min())
                        max_values.append(df.max())
                        mean_list.append(df.abs().mean())
                        RMS_list.append(np.sqrt(np.mean(np.square(df.to_numpy()))))
                        median_list.append(df.abs().median())
            
            subject_nr = subject_container.subject_nr
            #print('\n')
            print('Strong typing behavior, subject {}, minimum EMG value:'.format(subject_nr), min(min_values_sub))
            print('Strong typing behavior, subject {}, maximum EMG value:'.format(subject_nr), max(max_values_sub))
            print('Strong typing behavior, subject {}, mean EMG value:'.format(subject_nr), np.mean(mean_list_sub))
            print('Strong typing behavior, subject {}, RMS EMG value:'.format(subject_nr), np.sqrt(np.mean(np.square(RMS_list_sub))))
            print('Strong typing behavior, subject {}, median EMG value:'.format(subject_nr), np.median(median_list_sub))
            print('\n')

    else:
        raise Exception('Not available data type')
    
    print(min_values)
    print(max_values)
    print(mean_list)
    print(RMS_list)
    print(median_list)
    

# MAIN: ------------------------------------------------------------------------: 

if __name__ == "__main__":

    NR_SUBJECTS = 5
    NR_SESSIONS = 4

    soft_dir_name = 'Exp20201205_2myo_softType'
    hard_dir_name = 'Exp20201205_2myo_hardType'
    JSON_FILE_SOFT = 'mfcc_data_soft.json'
    JSON_FILE_HARD = 'mfcc_data_hard.json'

    csv_handler = CSV_handler(NR_SUBJECTS, NR_SESSIONS)
    dict = csv_handler.load_data('soft', soft_dir_name)
    

    
    

    #nn_handler = NN_handler(csv_handler)
    #nn_handler.store_mfcc_samples()
    #nn_handler.save_json_mfcc(JSON_FILE_SOFT)




