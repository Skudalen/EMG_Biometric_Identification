from Handle_emg_data.py import CSV_handler

def test_df_extraction():
    handler = CSV_handler()

    filename = "Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    subject1_left_emg1 = handler.get_time_emg_table(filename, 1)

    print(subject1_left_emg1.head())

test_df_extraction()
