import numpy as np 
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from scipy.fft import fft

import Handle_emg_data as Handler

SAMPLE_RATE = 200

def load_user_emg_data():

    # CSV data from subject 1
    file1_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    file2_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1830/myoLeftEmg.csv"
    file3_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1845/myoLeftEmg.csv"
    file4_subject1_left = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1855/myoLeftEmg.csv"
    subject1_left_files = [file1_subject1_left, file2_subject1_left, file3_subject1_left, file4_subject1_left]
    file1_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoRightEmg.csv"
    file2_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1830/myoRightEmg.csv"
    file3_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1845/myoRightEmg.csv"
    file4_subject1_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1855/myoRightEmg.csv"
    subject1_right_files = [file1_subject1_rigth, file2_subject1_rigth, file3_subject1_rigth, file4_subject1_rigth]

    # CSV data from subject 2
    file1_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2010/myoLeftEmg.csv"
    file2_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2025/myoLeftEmg.csv"
    file3_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2035/myoLeftEmg.csv"
    file4_subject2_left = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2045/myoLeftEmg.csv"
    subject2_left_files = [file1_subject2_left, file2_subject2_left, file3_subject2_left, file4_subject2_left]
    file1_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2010/myoRightEmg.csv"
    file2_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2025/myoRightEmg.csv"
    file3_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2035/myoRightEmg.csv"
    file4_subject2_rigth = "/Exp20201205_2myo_hardTypePP/HaluskaMaros_20201205_2045/myoRightEmg.csv"
    subject2_right_files = [file1_subject2_rigth, file2_subject2_rigth, file3_subject2_rigth, file4_subject2_rigth]

    # CSV data from subject 3
    file1_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1700/myoLeftEmg.csv"
    file2_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1715/myoLeftEmg.csv"
    file3_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1725/myoLeftEmg.csv"
    file4_subject3_left = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1735/myoLeftEmg.csv"
    subject3_left_files = [file1_subject3_left, file2_subject3_left, file3_subject3_left, file4_subject3_left]
    file1_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1700/myoRightEmg.csv"
    file2_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1715/myoRightEmg.csv"
    file3_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1725/myoRightEmg.csv"
    file4_subject3_rigth = "/Exp20201205_2myo_hardTypePP/HaluskovaBeata_20201205_1735/myoRightEmg.csv"
    subject3_right_files = [file1_subject3_rigth, file2_subject3_rigth, file3_subject3_rigth, file4_subject3_rigth]

    # CSV data from subject 4
    file1_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1900/myoLeftEmg.csv"
    file2_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1915/myoLeftEmg.csv"
    file3_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1925/myoLeftEmg.csv"
    file4_subject4_left = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1935/myoLeftEmg.csv"
    subject4_left_files = [file1_subject4_left, file2_subject4_left, file3_subject4_left, file4_subject4_left]
    file1_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1900/myoRightEmg.csv"
    file2_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1915/myoRightEmg.csv"
    file3_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1925/myoRightEmg.csv"
    file4_subject4_rigth = "/Exp20201205_2myo_hardTypePP/KelisekDavid_20201209_1935/myoRightEmg.csv"
    subject4_right_files = [file1_subject4_rigth, file2_subject4_rigth, file3_subject4_rigth, file4_subject4_rigth]


    # CSV data from subject 5
    file1_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2030/myoLeftEmg.csv"
    file2_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2040/myoLeftEmg.csv"
    file3_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2050/myoLeftEmg.csv"
    file4_subject5_left = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2100/myoLeftEmg.csv"
    subject5_left_files = [file1_subject5_left, file2_subject5_left, file3_subject5_left, file4_subject5_left]
    file1_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2030/myoRightEmg.csv"
    file2_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2040/myoRightEmg.csv"
    file3_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2050/myoRightEmg.csv"
    file4_subject5_rigth = "/Exp20201205_2myo_hardTypePP/KelisekRichard_20201209_2100/myoRightEmg.csv"
    subject5_right_files = [file1_subject5_rigth, file2_subject5_rigth, file3_subject5_rigth, file4_subject5_rigth]

    left_list = [subject1_left_files, subject2_left_files, subject3_left_files, subject4_left_files, subject5_left_files]
    right_list = [subject1_right_files, subject2_right_files, subject3_right_files, subject4_right_files, subject5_right_files]

    csv_handler = CSV_handler

    subject1_data_container = Data_container(1, 'HaluskaMarek')
    subject2_data_container = Data_container(1, 'HaluskaMaros')
    subject3_data_container = Data_container(1, 'HaluskovaBeata')
    subject4_data_container = Data_container(1, 'KelisekDavid')
    subject5_data_container = Data_container(1, 'KelisekRichard')
    subject_data_container_list = [subject1_data_container, subject2_data_container, subject3_data_container, 
                                    subject4_data_container, subject5_data_container]
    

    for subject_nr in range(5):
        # left variant proccessed here
        for round in range(4):
            for emg_nr in range(8):
                csv_handler.store_df(left_list[subject_nr][round], emg_nr+1, 'left', subject_data_container_list[subject_nr])
        # right variant proccessed here
        for round in range(4):
            for emg_nr in range(8):
                csv_handler.store_df(left_list[subject_nr][round], emg_nr+1, 'right', subject_data_container_list[subject_nr])

    return csv_handler.data_container_dict

def prep_df_for_trans(df:DataFrame):
    sample_rate = SAMPLE_RATE
    min, duration = Handler.get_min_max_timestamp(df)
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    y = np.array(df.iloc(1))
    return x, y

def normalize_wave(y_values):
    return None


def transformed_df(df:DataFrame):
    x, y = prep_df_for_trans(df)
    y_values = fft(y)
    return new_df

def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()


handler = CSV_handler
file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
df = handler.get_time_emg_table(file, 1)
#plot_df(df)
trans_df = DataFrame(transformed_df(df))
print(trans_df.info)
plot_df(trans_df)
