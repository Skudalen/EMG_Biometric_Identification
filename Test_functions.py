from Handle_emg_data import CSV_handler, get_min_max_timestamp
import matplotlib.pyplot as plt
import Signal_prep

def test_df_extraction(emg_nr):
    handler = CSV_handler()

    file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    subject1_left_emg1 = handler.get_time_emg_table(file, emg_nr)
    print(subject1_left_emg1.head)

    return subject1_left_emg1, emg_nr

def test_load_func():
    test_dict = Signal_prep.load_user_emg_data()
    subject2_container = test_dict[2]
    print(subject2_container.data_dict['left'][1])

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

    x, y, d = Signal_prep.prep_df_for_trans(df)
    print(x)
    print(y)
