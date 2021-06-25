from numpy import load
from Handle_emg_data import *
import matplotlib.pyplot as plt
from Signal_prep import *

def test_df_extraction(emg_nr):
    handler = CSV_handler()

    file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    subject1_left_emg1 = handler.get_time_emg_table(file, emg_nr)
    print(subject1_left_emg1.head)

    return subject1_left_emg1, emg_nr

def test_hardPP_load_func():
    handler = CSV_handler()
    test_dict = handler.load_hard_PP_emg_data()
    subject2_container = test_dict.get(2)
    print(subject2_container)
    print(subject2_container.subject_name)
    print(subject2_container.data_dict_round1.get('left')[1])

def test_min_max_func():
    handler = CSV_handler()
    file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    df = handler.get_time_emg_table(file, 1)
    min, max = get_min_max_timestamp(df)
    print(min)
    print(max)

def test_fft_prep():
    handler = CSV_handler()
    file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    df = handler.get_time_emg_table(file, 1)

def test_plot_wavelet_both_ways():
    handler = CSV_handler()
    file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    df = handler.get_time_emg_table(file, 1)
    N = get_xory_from_df('x', df)
    plot_df(df)
    #print(len(N))
    #print(len(get_xory_from_df('y', df)))
    x, cA, cD = wavelet_db4_denoising(df)
    plot_arrays(x, cA)
    #print(len(cA))
    cA_filt, cD_filt = soft_threshold_filter(cA, cD)
    plot_arrays(x, cA_filt)
    #print(len(cA_filt))
    y_new_values = inverse_wavelet(df, cA_filt, cD_filt)
    #print(len(y_new_values))
    plot_arrays(N, y_new_values)

def test_soft_load_func():
    handler = CSV_handler()
    test_dict = handler.load_soft_original_emg_data()
    subject2_container = test_dict.get(4)   # Subject 4
    print(subject2_container)
    print(subject2_container.subject_name)
    print(subject2_container.data_dict_round2.get('right')[3])  # Round2, right, emg_4

def test_total_denoising():
    handler = Handler.CSV_handler('hard')
    handler.load_hard_original_emg_data()

    # Original df:
    df = handler.get_df_from_data_dict(3, 'left', 3, 3)
    print(df.head)
    plot_df(df)

    # Denoised df:
    N_trans, y_values, df_denoised = denoice_dataset(handler, 3, 'left', 3, 3)
    print(df_denoised.head)
    plot_df(df_denoised)

test_total_denoising()
