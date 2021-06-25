import pandas as pd
from pathlib import Path
import numpy as np

from pandas.core.frame import DataFrame

class Data_container:
      
    def __init__(self, subject_nr:int, subject_name:str):
        self.subject_nr = subject_nr
        self.subject_name = subject_name
        self.data_dict_round1 = {'left': [None]*8, 'right': [None]*8}
        self.data_dict_round2 = {'left': [None]*8, 'right': [None]*8}
        self.data_dict_round3 = {'left': [None]*8, 'right': [None]*8}
        self.data_dict_round4 = {'left': [None]*8, 'right': [None]*8}
        self.dict_list =    [self.data_dict_round1, 
                            self.data_dict_round2, 
                            self.data_dict_round3, 
                            self.data_dict_round4
                            ]
    
class CSV_handler:

    def __init__(self, data_type):
        self.working_dir = str(Path.cwd()) 
        self.data_container_dict = {} # Dict with keys equal subject numbers and values equal the relvant datacontainer
        self.data_type = data_type

    # Makes dataframe from the csv files in the working directory
    def make_df(self, filename):
        filepath = self.working_dir + str(filename)
        df = pd.read_csv(filepath)
        return df

    # Extracts out the timestamp and the selected emg signal into a new dataframe and stores the data on the subject
    def get_time_emg_table(self, filename:str, emg_nr:int):
        tot_data_frame = self.make_df(filename)
        emg_str = 'emg' + str(emg_nr)
        filtered_df = tot_data_frame[["timestamp", emg_str]]
        return filtered_df
    
    # Takes in a df and stores the information in a Data_container object
    def store_df_in_container(self, filename:str, emg_nr:int, which_arm:str, data_container:Data_container, round:int):
        df = self.get_time_emg_table(filename, emg_nr+1)

        # Places the data correctly:
        if round == 1:
            if which_arm == 'left':
                data_container.data_dict_round1['left'][emg_nr] = df    # Zero indexed emg_nr in the dict
            else:
                data_container.data_dict_round1['right'][emg_nr] = df
        elif round == 2:
            if which_arm == 'left':
                data_container.data_dict_round2['left'][emg_nr] = df
            else:
                data_container.data_dict_round2['right'][emg_nr] = df
        elif round == 3:
            if which_arm == 'left':
                data_container.data_dict_round3['left'][emg_nr] = df
            else:
                data_container.data_dict_round3['right'][emg_nr] = df
        elif round == 4:
            if which_arm == 'left':
                data_container.data_dict_round4['left'][emg_nr] = df
            else:
                data_container.data_dict_round4['right'][emg_nr] = df
        else:
            raise IndexError('Not a valid index')
    
    # Links the data container for a subject to the handler object
    def link_container_to_handler(self, data_container:Data_container):
        # Links the retrieved data with the subjects data_container
        subject_nr = data_container.subject_nr
        self.data_container_dict[subject_nr] = data_container
    
    # Loads the data from the csv files into a storing system in an CSV_handler object
    # (hard, hardPP, soft and softPP)
    def load_hard_PP_emg_data(self):

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


        subject1_data_container = Data_container(1, 'HaluskaMarek')
        subject2_data_container = Data_container(2, 'HaluskaMaros')
        subject3_data_container = Data_container(3, 'HaluskovaBeata')
        subject4_data_container = Data_container(4, 'KelisekDavid')
        subject5_data_container = Data_container(5, 'KelisekRichard')
        subject_data_container_list = [subject1_data_container, subject2_data_container, subject3_data_container, 
                                        subject4_data_container, subject5_data_container]
    
        for subject_nr in range(5):
            data_container = subject_data_container_list[subject_nr]
            # left variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = left_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'left', data_container, round+1)
            # right variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = right_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'right', data_container, round+1)
            # Links the stored data in the data_container to the Handler
            self.link_container_to_handler(data_container)
        self.data_type = 'hardPP'
        return self.data_container_dict

    def load_soft_PP_emg_data(self):

        # CSV data from subject 1
        file1_subject1_left = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1910/myoLeftEmg.csv"
        file2_subject1_left = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1920/myoLeftEmg.csv"
        file3_subject1_left = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1935/myoLeftEmg.csv"
        file4_subject1_left = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1945/myoLeftEmg.csv"
        subject1_left_files = [file1_subject1_left, file2_subject1_left, file3_subject1_left, file4_subject1_left]
        file1_subject1_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1910/myoRightEmg.csv"
        file2_subject1_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1920/myoRightEmg.csv"
        file3_subject1_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1935/myoRightEmg.csv"
        file4_subject1_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMarek_20201207_1945/myoRightEmg.csv"
        subject1_right_files = [file1_subject1_rigth, file2_subject1_rigth, file3_subject1_rigth, file4_subject1_rigth]

        # CSV data from subject 2
        file1_subject2_left = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2055/myoLeftEmg.csv"
        file2_subject2_left = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2110/myoLeftEmg.csv"
        file3_subject2_left = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2125/myoLeftEmg.csv"
        file4_subject2_left = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2145/myoLeftEmg.csv"
        subject2_left_files = [file1_subject2_left, file2_subject2_left, file3_subject2_left, file4_subject2_left]
        file1_subject2_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2055/myoRightEmg.csv"
        file2_subject2_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2110/myoRightEmg.csv"
        file3_subject2_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2125/myoRightEmg.csv"
        file4_subject2_rigth = "/Exp20201205_2myo_softTypePP/HaluskaMaros_20201205_2145/myoRightEmg.csv"
        subject2_right_files = [file1_subject2_rigth, file2_subject2_rigth, file3_subject2_rigth, file4_subject2_rigth]

        # CSV data from subject 3
        file1_subject3_left = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1745/myoLeftEmg.csv"
        file2_subject3_left = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1755/myoLeftEmg.csv"
        file3_subject3_left = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1810/myoLeftEmg.csv"
        file4_subject3_left = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1825/myoLeftEmg.csv"
        subject3_left_files = [file1_subject3_left, file2_subject3_left, file3_subject3_left, file4_subject3_left]
        file1_subject3_rigth = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1745/myoRightEmg.csv"
        file2_subject3_rigth = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1755/myoRightEmg.csv"
        file3_subject3_rigth = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1810/myoRightEmg.csv"
        file4_subject3_rigth = "/Exp20201205_2myo_softTypePP/HaluskovaBeata_20201205_1825/myoRightEmg.csv"
        subject3_right_files = [file1_subject3_rigth, file2_subject3_rigth, file3_subject3_rigth, file4_subject3_rigth]

        # CSV data from subject 4
        file1_subject4_left = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_1945/myoLeftEmg.csv"
        file2_subject4_left = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_1955/myoLeftEmg.csv"
        file3_subject4_left = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_2010/myoLeftEmg.csv"
        file4_subject4_left = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_2025/myoLeftEmg.csv"
        subject4_left_files = [file1_subject4_left, file2_subject4_left, file3_subject4_left, file4_subject4_left]
        file1_subject4_rigth = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_1945/myoRightEmg.csv"
        file2_subject4_rigth = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_1955/myoRightEmg.csv"
        file3_subject4_rigth = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_2010/myoRightEmg.csv"
        file4_subject4_rigth = "/Exp20201205_2myo_softTypePP/KelisekDavid_20201209_2025/myoRightEmg.csv"
        subject4_right_files = [file1_subject4_rigth, file2_subject4_rigth, file3_subject4_rigth, file4_subject4_rigth]


        # CSV data from subject 5
        file1_subject5_left = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2110/myoLeftEmg.csv"
        file2_subject5_left = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2120/myoLeftEmg.csv"
        file3_subject5_left = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2130/myoLeftEmg.csv"
        file4_subject5_left = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2140/myoLeftEmg.csv"
        subject5_left_files = [file1_subject5_left, file2_subject5_left, file3_subject5_left, file4_subject5_left]
        file1_subject5_rigth = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2110/myoRightEmg.csv"
        file2_subject5_rigth = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2120/myoRightEmg.csv"
        file3_subject5_rigth = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2130/myoRightEmg.csv"
        file4_subject5_rigth = "/Exp20201205_2myo_softTypePP/KelisekRichard_20201209_2140/myoRightEmg.csv"
        subject5_right_files = [file1_subject5_rigth, file2_subject5_rigth, file3_subject5_rigth, file4_subject5_rigth]

        left_list = [subject1_left_files, subject2_left_files, subject3_left_files, subject4_left_files, subject5_left_files]
        right_list = [subject1_right_files, subject2_right_files, subject3_right_files, subject4_right_files, subject5_right_files]


        subject1_data_container = Data_container(1, 'HaluskaMarek')
        subject2_data_container = Data_container(2, 'HaluskaMaros')
        subject3_data_container = Data_container(3, 'HaluskovaBeata')
        subject4_data_container = Data_container(4, 'KelisekDavid')
        subject5_data_container = Data_container(5, 'KelisekRichard')
        subject_data_container_list = [subject1_data_container, subject2_data_container, subject3_data_container, 
                                        subject4_data_container, subject5_data_container]
    
        for subject_nr in range(5):
            data_container = subject_data_container_list[subject_nr]
            # left variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = left_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'left', data_container, round+1)
            # right variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = right_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'right', data_container, round+1)
            # Links the stored data in the data_container to the Handler
            self.link_container_to_handler(data_container)
        self.data_type = 'softPP'
        return self.data_container_dict

    def load_hard_original_emg_data(self):

         # CSV data from subject 1
        file1_subject1_left = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
        file2_subject1_left = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1830/myoLeftEmg.csv"
        file3_subject1_left = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1845/myoLeftEmg.csv"
        file4_subject1_left = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1855/myoLeftEmg.csv"
        subject1_left_files = [file1_subject1_left, file2_subject1_left, file3_subject1_left, file4_subject1_left]
        file1_subject1_rigth = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1810/myoRightEmg.csv"
        file2_subject1_rigth = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1830/myoRightEmg.csv"
        file3_subject1_rigth = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1845/myoRightEmg.csv"
        file4_subject1_rigth = "/Exp20201205_2myo_hardType/HaluskaMarek_20201207_1855/myoRightEmg.csv"
        subject1_right_files = [file1_subject1_rigth, file2_subject1_rigth, file3_subject1_rigth, file4_subject1_rigth]

        # CSV data from subject 2
        file1_subject2_left = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2010/myoLeftEmg.csv"
        file2_subject2_left = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2025/myoLeftEmg.csv"
        file3_subject2_left = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2035/myoLeftEmg.csv"
        file4_subject2_left = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2045/myoLeftEmg.csv"
        subject2_left_files = [file1_subject2_left, file2_subject2_left, file3_subject2_left, file4_subject2_left]
        file1_subject2_rigth = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2010/myoRightEmg.csv"
        file2_subject2_rigth = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2025/myoRightEmg.csv"
        file3_subject2_rigth = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2035/myoRightEmg.csv"
        file4_subject2_rigth = "/Exp20201205_2myo_hardType/HaluskaMaros_20201205_2045/myoRightEmg.csv"
        subject2_right_files = [file1_subject2_rigth, file2_subject2_rigth, file3_subject2_rigth, file4_subject2_rigth]

        # CSV data from subject 3
        file1_subject3_left = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1700/myoLeftEmg.csv"
        file2_subject3_left = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1715/myoLeftEmg.csv"
        file3_subject3_left = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1725/myoLeftEmg.csv"
        file4_subject3_left = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1735/myoLeftEmg.csv"
        subject3_left_files = [file1_subject3_left, file2_subject3_left, file3_subject3_left, file4_subject3_left]
        file1_subject3_rigth = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1700/myoRightEmg.csv"
        file2_subject3_rigth = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1715/myoRightEmg.csv"
        file3_subject3_rigth = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1725/myoRightEmg.csv"
        file4_subject3_rigth = "/Exp20201205_2myo_hardType/HaluskovaBeata_20201205_1735/myoRightEmg.csv"
        subject3_right_files = [file1_subject3_rigth, file2_subject3_rigth, file3_subject3_rigth, file4_subject3_rigth]

        # CSV data from subject 4
        file1_subject4_left = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1900/myoLeftEmg.csv"
        file2_subject4_left = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1915/myoLeftEmg.csv"
        file3_subject4_left = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1925/myoLeftEmg.csv"
        file4_subject4_left = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1935/myoLeftEmg.csv"
        subject4_left_files = [file1_subject4_left, file2_subject4_left, file3_subject4_left, file4_subject4_left]
        file1_subject4_rigth = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1900/myoRightEmg.csv"
        file2_subject4_rigth = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1915/myoRightEmg.csv"
        file3_subject4_rigth = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1925/myoRightEmg.csv"
        file4_subject4_rigth = "/Exp20201205_2myo_hardType/KelisekDavid_20201209_1935/myoRightEmg.csv"
        subject4_right_files = [file1_subject4_rigth, file2_subject4_rigth, file3_subject4_rigth, file4_subject4_rigth]


        # CSV data from subject 5
        file1_subject5_left = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2030/myoLeftEmg.csv"
        file2_subject5_left = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2040/myoLeftEmg.csv"
        file3_subject5_left = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2050/myoLeftEmg.csv"
        file4_subject5_left = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2100/myoLeftEmg.csv"
        subject5_left_files = [file1_subject5_left, file2_subject5_left, file3_subject5_left, file4_subject5_left]
        file1_subject5_rigth = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2030/myoRightEmg.csv"
        file2_subject5_rigth = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2040/myoRightEmg.csv"
        file3_subject5_rigth = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2050/myoRightEmg.csv"
        file4_subject5_rigth = "/Exp20201205_2myo_hardType/KelisekRichard_20201209_2100/myoRightEmg.csv"
        subject5_right_files = [file1_subject5_rigth, file2_subject5_rigth, file3_subject5_rigth, file4_subject5_rigth]

        left_list = [subject1_left_files, subject2_left_files, subject3_left_files, subject4_left_files, subject5_left_files]
        right_list = [subject1_right_files, subject2_right_files, subject3_right_files, subject4_right_files, subject5_right_files]


        subject1_data_container = Data_container(1, 'HaluskaMarek')
        subject2_data_container = Data_container(2, 'HaluskaMaros')
        subject3_data_container = Data_container(3, 'HaluskovaBeata')
        subject4_data_container = Data_container(4, 'KelisekDavid')
        subject5_data_container = Data_container(5, 'KelisekRichard')
        subject_data_container_list = [subject1_data_container, subject2_data_container, subject3_data_container, 
                                        subject4_data_container, subject5_data_container]
    
        for subject_nr in range(5):
            data_container = subject_data_container_list[subject_nr]
            # left variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = left_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'left', data_container, round+1)
            # right variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = right_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'right', data_container, round+1)
            # Links the stored data in the data_container to the Handler
            self.link_container_to_handler(data_container)
        self.data_type = 'hard'
        return self.data_container_dict

    def load_soft_original_emg_data(self):

        # CSV data from subject 1
        file1_subject1_left = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1910/myoLeftEmg.csv"
        file2_subject1_left = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1920/myoLeftEmg.csv"
        file3_subject1_left = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1935/myoLeftEmg.csv"
        file4_subject1_left = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1945/myoLeftEmg.csv"
        subject1_left_files = [file1_subject1_left, file2_subject1_left, file3_subject1_left, file4_subject1_left]
        file1_subject1_rigth = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1910/myoRightEmg.csv"
        file2_subject1_rigth = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1920/myoRightEmg.csv"
        file3_subject1_rigth = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1935/myoRightEmg.csv"
        file4_subject1_rigth = "/Exp20201205_2myo_softType/HaluskaMarek_20201207_1945/myoRightEmg.csv"
        subject1_right_files = [file1_subject1_rigth, file2_subject1_rigth, file3_subject1_rigth, file4_subject1_rigth]

        # CSV data from subject 2
        file1_subject2_left = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2055/myoLeftEmg.csv"
        file2_subject2_left = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2110/myoLeftEmg.csv"
        file3_subject2_left = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2125/myoLeftEmg.csv"
        file4_subject2_left = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2145/myoLeftEmg.csv"
        subject2_left_files = [file1_subject2_left, file2_subject2_left, file3_subject2_left, file4_subject2_left]
        file1_subject2_rigth = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2055/myoRightEmg.csv"
        file2_subject2_rigth = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2110/myoRightEmg.csv"
        file3_subject2_rigth = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2125/myoRightEmg.csv"
        file4_subject2_rigth = "/Exp20201205_2myo_softType/HaluskaMaros_20201205_2145/myoRightEmg.csv"
        subject2_right_files = [file1_subject2_rigth, file2_subject2_rigth, file3_subject2_rigth, file4_subject2_rigth]

        # CSV data from subject 3
        file1_subject3_left = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1745/myoLeftEmg.csv"
        file2_subject3_left = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1755/myoLeftEmg.csv"
        file3_subject3_left = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1810/myoLeftEmg.csv"
        file4_subject3_left = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1825/myoLeftEmg.csv"
        subject3_left_files = [file1_subject3_left, file2_subject3_left, file3_subject3_left, file4_subject3_left]
        file1_subject3_rigth = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1745/myoRightEmg.csv"
        file2_subject3_rigth = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1755/myoRightEmg.csv"
        file3_subject3_rigth = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1810/myoRightEmg.csv"
        file4_subject3_rigth = "/Exp20201205_2myo_softType/HaluskovaBeata_20201205_1825/myoRightEmg.csv"
        subject3_right_files = [file1_subject3_rigth, file2_subject3_rigth, file3_subject3_rigth, file4_subject3_rigth]

        # CSV data from subject 4
        file1_subject4_left = "/Exp20201205_2myo_softType/KelisekDavid_20201209_1945/myoLeftEmg.csv"
        file2_subject4_left = "/Exp20201205_2myo_softType/KelisekDavid_20201209_1955/myoLeftEmg.csv"
        file3_subject4_left = "/Exp20201205_2myo_softType/KelisekDavid_20201209_2010/myoLeftEmg.csv"
        file4_subject4_left = "/Exp20201205_2myo_softType/KelisekDavid_20201209_2025/myoLeftEmg.csv"
        subject4_left_files = [file1_subject4_left, file2_subject4_left, file3_subject4_left, file4_subject4_left]
        file1_subject4_rigth = "/Exp20201205_2myo_softType/KelisekDavid_20201209_1945/myoRightEmg.csv"
        file2_subject4_rigth = "/Exp20201205_2myo_softType/KelisekDavid_20201209_1955/myoRightEmg.csv"
        file3_subject4_rigth = "/Exp20201205_2myo_softType/KelisekDavid_20201209_2010/myoRightEmg.csv"
        file4_subject4_rigth = "/Exp20201205_2myo_softType/KelisekDavid_20201209_2025/myoRightEmg.csv"
        subject4_right_files = [file1_subject4_rigth, file2_subject4_rigth, file3_subject4_rigth, file4_subject4_rigth]


        # CSV data from subject 5
        file1_subject5_left = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2110/myoLeftEmg.csv"
        file2_subject5_left = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2120/myoLeftEmg.csv"
        file3_subject5_left = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2130/myoLeftEmg.csv"
        file4_subject5_left = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2140/myoLeftEmg.csv"
        subject5_left_files = [file1_subject5_left, file2_subject5_left, file3_subject5_left, file4_subject5_left]
        file1_subject5_rigth = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2110/myoRightEmg.csv"
        file2_subject5_rigth = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2120/myoRightEmg.csv"
        file3_subject5_rigth = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2130/myoRightEmg.csv"
        file4_subject5_rigth = "/Exp20201205_2myo_softType/KelisekRichard_20201209_2140/myoRightEmg.csv"
        subject5_right_files = [file1_subject5_rigth, file2_subject5_rigth, file3_subject5_rigth, file4_subject5_rigth]

        left_list = [subject1_left_files, subject2_left_files, subject3_left_files, subject4_left_files, subject5_left_files]
        right_list = [subject1_right_files, subject2_right_files, subject3_right_files, subject4_right_files, subject5_right_files]


        subject1_data_container = Data_container(1, 'HaluskaMarek')
        subject2_data_container = Data_container(2, 'HaluskaMaros')
        subject3_data_container = Data_container(3, 'HaluskovaBeata')
        subject4_data_container = Data_container(4, 'KelisekDavid')
        subject5_data_container = Data_container(5, 'KelisekRichard')
        subject_data_container_list = [subject1_data_container, subject2_data_container, subject3_data_container, 
                                        subject4_data_container, subject5_data_container]
    
        for subject_nr in range(5):
            data_container = subject_data_container_list[subject_nr]
            # left variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = left_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'left', data_container, round+1)
            # right variant proccessed here
            for round in range(4):
                for emg_nr in range(8):
                    filename = right_list[subject_nr][round]
                    self.store_df_in_container(filename, emg_nr, 'right', data_container, round+1)
            # Links the stored data in the data_container to the Handler
            self.link_container_to_handler(data_container)
        self.data_type = 'soft'
        return self.data_container_dict

    # Retrieves df via the data_dict in the handler object
    def get_df_from_data_dict(self, subject_nr, which_arm, round, emg_nr):
        data_type = self.data_type
        container = self.data_container_dict.get(subject_nr)
        df = container.dict_list[round - 1].get(which_arm)[emg_nr]
        return df


# Help: gets the str from emg nr
def get_emg_str(emg_nr):
    return 'emg' + str(emg_nr)

# Help: gets the min/max of a df
def get_min_max_timestamp(df:DataFrame):
    min = int(np.floor(df['timestamp'].min()))
    max = df['timestamp'].max()
    return min, max


