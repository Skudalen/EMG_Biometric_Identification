from Handle_emg_data import CSV_handler
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

