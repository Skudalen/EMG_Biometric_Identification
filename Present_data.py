from Handle_emg_data import *
from Signal_prep import *

# PLOT FUNCTIONS:

# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
def plot_arrays(N, N_name, y, y_name):
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
    
#Takes in handler and detailes to denoise. Returns arrays and df
def denoice_dataset(handler:Handler.CSV_handler, subject_nr, which_arm, round, emg_nr):
    df = handler.get_df_from_data_dict(subject_nr, which_arm, round, emg_nr)

    N = get_xory_from_df('x', df)
    N_trans, cA, cD = wavelet_db4(df)
    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    y_values = inverse_wavelet(df, cA_filt, cD_filt)

    df_new = Handler.make_df_from_xandy(N, y_values, emg_nr)
    return df_new



# MAIN:

def main():

    csv_handler = CSV_handler('hard')
    load_data(csv_handler, 'hard')
    data_frame = get_data(csv_handler, 1, 'left', 1, 1)

    N_trans, cA, cD = wavelet_db4(data_frame)
    data_frame_freq = make_df_from_xandy(N_trans, cA, 1)

    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    data_frame_freq_filt = make_df_from_xandy(N_trans, cD_filt, 1)

    plot_compare_two_df(data_frame_freq, 'Original data', data_frame_freq_filt, 'Analyzed data')

    return None

main()