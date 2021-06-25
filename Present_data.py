from Handle_emg_data import *
from Signal_prep import *

# PLOT FUNCTIONS:

# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
def plot_arrays(N, y):
    plt.plot(N, np.abs(y))
    plt.show()

def plot_compare_two_df(df_old, df_new):
    x = get_xory_from_df('x', df_old)
    y1 = get_xory_from_df('y', df_old)
    y2 = get_xory_from_df('y', df_new)

    figure, axis = plt.subplots(1, 2)
    axis[0].plot(x, y1)
    axis[0].set_title('Original data')
    axis[1].plot(x, y2)
    axis[1].set_title('Analyzed data')
    plt.show()


# DATA FUNCTIONS:

# The CSV_handler takes in data_type, but only for visuals. 
# E.g. handler = CSV_handler('soft')

# Loads in data. Choose data_type: hard, hardPP, soft og softPP as str. Returns None
def load_data(csv_handler:CSV_handler, data_type):
    switcher = {
                'hard': csv_handler.load_hard_original_emg_data(),
                'hardPP':csv_handler.load_hard_PP_emg_data(),
                'soft':csv_handler.load_soft_original_emg_data(),
                'softPP':csv_handler.load_soft_PP_emg_data(),
                }
    return switcher.get(data_type)

# Retrieved data. Send in loaded csv_handler and data detailes you want. Returns DataFrame
def get_data(csv_handler:CSV_handler, subject_nr, which_arm, session, emg_nr):
    data_frame = csv_handler.get_df_from_data_dict(subject_nr, which_arm, session, emg_nr)
    return data_frame
    

# MAIN:

def main():

    csv_handler = CSV_handler('hard')
    load_data(csv_handler, 'hard')
    data_frame = get_data(csv_handler, 1, 'left', 1, 1)

    N_trans, cA, cD = wavelet_db4_denoising(data_frame)
    data_frame_freq = make_df_from_xandy(N_trans, cA, 1)

    cA_filt, cD_filt = soft_threshold_filter(cA, cD, 0.4)
    data_frame_freq_filt = make_df_from_xandy(N_trans, cA_filt, 1)

    plot_compare_two_df(data_frame_freq, data_frame_freq_filt)

    return None

main()