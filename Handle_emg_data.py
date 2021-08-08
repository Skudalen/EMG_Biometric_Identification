from typing import List
from numpy.core.arrayprint import IntegerFormat
from numpy.lib import math
import pandas as pd
from pathlib import Path
import numpy as np
from pandas.core.frame import DataFrame
import sys
sys.path.insert(0, '/Users/Markus/Prosjekter git/Slovakia 2021/psf_lib/python_speech_features/python_speech_features')
from psf_lib.python_speech_features.python_speech_features import mfcc
import json
import os


# Global variables for MFCC
MFCC_STEPSIZE = 0.5     # Seconds
MFCC_WINDOWSIZE = 2     # Seconds
NR_COEFFICIENTS = 13    # Number of coefficients
NR_MEL_BINS = 40     # Number of mel-filter-bins 

class Data_container:
    
    # Initiates personal data container for each subject. Dict for each session with keys 'left' and 'right',
    # and values equal to lists of EMG data indexed 0-7  
    def __init__(self, subject_nr:int, subject_name:str, nr_sessions:int):
        self.subject_nr = subject_nr
        self.subject_name = subject_name
        self.dict_list =  [{'left': [None]*8, 'right': [None]*8} for i in range(nr_sessions)]

    def __str__(self) -> str:
        return 'Name: {}, \tID: {}'.format(self.subject_name, self.subject_nr)

class CSV_handler:

    # Initiates object to store all datapoints in the experiment
    def __init__(self, nr_subjects:int, nr_sessions:int):
        self.working_dir = str(Path.cwd())
        self.nr_subjects = nr_subjects
        self.nr_sessions = nr_sessions
        # Dict with keys equal subject numbers and values equal to its respective datacontainer 
        self.data_container_dict = {i+1: None for i in range(nr_subjects)}  
        # String describing which type of data is stored in the object 
        self.data_type = None   

    # Makes dataframe from the csv files in the working directory
    # Input: filename of a csv-file
    # Output: DataFrame
    def make_df(self, filename):
        #filepath = self.working_dir + str(filename)
        filepath = str(filename)
        df = pd.read_csv(filepath)
        return df

    # Extracts out the timestamp and the selected emg signal into a new dataframe
    # Input: filename of a csv-file, EMG nr 
    # Output: DataFrame(timestamp/EMG)
    def get_emg_table_from_file(self, filename:str, emg_nr:int):
        tot_data_frame = self.make_df(filename)
        emg_str = 'emg' + str(emg_nr)
        filtered_df = tot_data_frame[["timestamp", emg_str]]
        return filtered_df
    
    # Takes in a df and stores the information in a Data_container object
    # Input: filename of a csv-file, EMG nr, left/right arm, subject's data_container, session nr 
    # Output: None -> stores EMG data in data container
    def store_df_in_container(self, filename:str, emg_nr:int, which_arm:str, data_container:Data_container, session:int):
        df = self.get_emg_table_from_file(filename, emg_nr+1)

        if df.isnull().values.any():
            print('NaN in: subject', data_container.subject_nr, 'arm:', which_arm, 'session:', session, 'emg nr:', emg_nr)

        # Places the data correctly:
        data_container.dict_list[session-1][which_arm][emg_nr] = df

    
    # Links the data container for a subject to the csv_handler object
    # Input: the subject's data_container
    # Output: None -> places the data container correctly in the CSV_handler data_container_dict
    def link_container_to_handler(self, data_container:Data_container):
        # Links the retrieved data with the subjects data_container
        subject_nr = data_container.subject_nr
        self.data_container_dict[subject_nr] = data_container

    # Retrieves df via the data_dict in the CSV_handler object  
    # Input: Experiment detailes
    # Output: DataFrame
    def get_df_from_data_dict(self, subject_nr, which_arm, session, emg_nr):
        container:Data_container = self.data_container_dict.get(subject_nr)
        df = container.dict_list[session - 1].get(which_arm)[emg_nr - 1]
        return df


    # Loads data the to the CSV_handler(general load func). Choose data_type: hard, hardPP, soft og softPP as str. 
    # Input: String(datatype you want), direction name of that type
    # Output: None -> load and stores data
    def load_data(self, type:str, type_dir_name:str):

        data_path = self.working_dir + '/data/' + type_dir_name
        subject_id = 100
        subject_name = 'bruh'
        nr_sessions = 101
        container = None
        session_count = 0

        for i, (path, subject_dir, session_dir) in enumerate(os.walk(data_path)):
            
            if path is not data_path:
                
                if subject_dir:
                    session_count = 0
                    subject_id = int(path[-1])
                    subject_name = subject_dir[0].split('_')[0]
                    nr_sessions = len(subject_dir)
                    container = Data_container(subject_id, subject_name, nr_sessions)
                    continue
                else:
                    session_count += 1

                for f in session_dir:
                    spes_path = os.path.join(path, f)
                    if f == 'myoLeftEmg.csv':
                        for emg_nr in range(8):
                            self.store_df_in_container(spes_path, emg_nr, 'left', container, session_count)
                    elif f == 'myoRightEmg.csv':
                        for emg_nr in range(8):
                            self.store_df_in_container(spes_path, emg_nr, 'right', container, session_count)
                self.link_container_to_handler(container)
        self.data_type = type
        return self.data_container_dict

    # Retrieved data. Send in loaded csv_handler and data detailes you want. 
    # Input: Experiment detailes
    # Output: DataFrame,  samplerate:int
    def get_data(self, subject_nr, which_arm, session, emg_nr):
        data_frame = self.get_df_from_data_dict(subject_nr, which_arm, session, emg_nr)
        samplerate = get_samplerate(data_frame)
        return data_frame, samplerate


    # OBSOLETE
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
    def load_data_OLD(self, data_type):
        if data_type == 'hard': 
            self.load_hard_original_emg_data()
        elif data_type == 'hardPP':
            self.load_hard_PP_emg_data()
        elif data_type == 'soft':
            self.load_soft_original_emg_data()
        elif data_type == 'softPP':
            self.load_soft_PP_emg_data()
        else:
            raise Exception('Wrong input')


    # NOT IMPLEMENTED
    def get_keyboard_data(self, filename:str, pres_or_release:str='pressed'):
        filepath = self.working_dir + str(filename)
        df = pd.read_csv(filepath)
        if pres_or_release == 'pressed':
            df = df[(df['event'] == 'KeyPressed') and (df['event'] == 'KeyPressed')]
        else: pass
        pass


class NN_handler:

    # Paths for data storage in json to later use in Neural_Network_Analysis.py
    JSON_PATH_MFCC = "mfcc_data.json"

    # Class to manipulate data from the CSV_handler and store it for further analysis
    def __init__(self, csv_handler:CSV_handler) -> None:
        self.csv_handler = csv_handler
        # Should med 4 sessions * (~150, 208) of mfcc samples per person. One [DataFrame, session_length_list] per subject
        self.mfcc_samples_per_subject = {k+1:[] for k in range(csv_handler.nr_subjects)}

    # GET method for mfcc_samples_dict  
    def get_mfcc_samples_dict(self) -> dict:
        return self.mfcc_samples_per_subject

    # Retrieves all EMG data from one subject and one session, and makes a list of the DataFrames
    # Input: Subject nr, Session nr (norm, not 0-indexed)
    # Output: List(df_1, ..., df_16)
    def get_emg_list(self, subject_nr, session_nr) -> list:
        list_of_emgs = []
        df, _ = self.csv_handler.get_data(subject_nr, 'left', session_nr, 1)
        list_of_emgs.append(df)
        for emg_nr in range(7):
            df, _ = self.csv_handler.get_data(subject_nr, 'left', session_nr, emg_nr+2)
            list_of_emgs.append(DataFrame(df[get_emg_str(emg_nr+2)]))
        for emg_nr in range(8):
            df, _ = self.csv_handler.get_data(subject_nr, 'right', session_nr, emg_nr+1)
            list_of_emgs.append(DataFrame(df[get_emg_str(emg_nr+1)]))
        
        return list_of_emgs     # list of emg data 

    # Creates one Dataframe of all EMG data(one session, one subject). One column for each EMG array
    # Input: List(emg1, ..., emg16)
    # Output: DataFrame(shape[1]=16)
    def make_subj_sample(self, list_of_emgs_):
        # Test and fix if the left/right EMGs have different size
        list_of_emgs = []
        length_left_emgs = int(len(list_of_emgs_[0].index))
        length_right_emgs = int(len(list_of_emgs_[-1].index))
        if length_left_emgs < length_right_emgs: 
            for i in range(16):
                new_emg_df = list_of_emgs_[i].head(length_left_emgs)
                list_of_emgs.append(new_emg_df)
        elif length_right_emgs < length_left_emgs:
            for i in range(16):
                new_emg_df = list_of_emgs_[i].head(length_right_emgs)
                list_of_emgs.append(new_emg_df)
        else:
            list_of_emgs = list_of_emgs_
        
        tot_session_df_list = []
        for i in range(8):
            df = list_of_emgs[i]
            tot_session_df_list.append(df)
        for i in range(1, 9):
            emg_str_old = get_emg_str(i)
            emg_str_new = get_emg_str(8+i)
            df = list_of_emgs[7+i].rename(columns={emg_str_old: emg_str_new})
            tot_session_df_list.append(df)
        tot_session_df = pd.concat(tot_session_df_list, axis=1, ignore_index=True)

        return tot_session_df
    
    # Takes in all EMG session Dataframe and creates DataFrame of MFCC samples
    # Input: DataFrame(shape[1]=16, EMG data)
    # Output: DataFrame(merged MFCC data, shape: (n, 13*16)), length of session datapoints
    def make_mfcc_df_from_session_df(self, session_df):
        session_df.rename(columns={0:'timestamp'}, inplace=True)
        samplerate = get_samplerate(session_df)
        attach_func = lambda list_1, list_2: list_1.extend(list_2)

        signal = session_df[1]
        mfcc_0 = mfcc_custom(signal, samplerate, MFCC_WINDOWSIZE, MFCC_STEPSIZE, NR_COEFFICIENTS, NR_MEL_BINS)
        df = DataFrame(mfcc_0).dropna()
        df['combined'] = df.values.tolist()
        result_df = df['combined']

        for i in range(2, 17):
            signal_i = session_df[i]
            mfcc_i = mfcc_custom(signal_i, samplerate, MFCC_WINDOWSIZE, MFCC_STEPSIZE, NR_COEFFICIENTS, NR_MEL_BINS)
            mfcc_i = DataFrame(mfcc_i).dropna()
            mfcc_i['combined'] = mfcc_i.values.tolist()
            df = result_df.combine(mfcc_i['combined'], attach_func)
        
        session_length = (len(result_df.index)) # Add the length of session data points

        return result_df, session_length

    # Merges MFCC data from all sessions and stores the sample data in 
    # the NN_handler's mfcc_samples_per_subject dict
    # Input: None(NN_handler)
    # Output: None -> stores in NN_handler [samples, session_length_list] for each subject
    def store_mfcc_samples(self) -> None:
        for subject_nr in range(5):
            subj_samples = []
            session_length_list = []
            for session_nr in range(4):
                list_of_emg = self.get_emg_list(subject_nr+1, session_nr+1)
                tot_session_df = self.make_subj_sample(list_of_emg)
            
                # TESTING FOR NAN
                if tot_session_df.isnull().values.any():
                    print('NaN in: subject', subject_nr+1, 'session:', session_nr+1, 'where? HERE')
                
                mfcc_df_i, session_length = self.make_mfcc_df_from_session_df(tot_session_df)
                subj_samples.append(mfcc_df_i)
                session_length_list.append(session_length)            
            
            result_df = pd.concat(subj_samples, axis=0, ignore_index=True)
            self.mfcc_samples_per_subject[subject_nr+1] = [result_df, session_length_list]

    # Stores MFCC data from mfcc_samples_per_subject in a json file
    # Input: Path to the json file
    # Output: None -> stores in json
    def save_json_mfcc(self, json_path=JSON_PATH_MFCC):
            
            # dictionary to store mapping, labels, and MFCCs
            data = {
                "mapping": [],
                "labels": [],
                "mfcc": [],

                "session_lengths": []
            }

            raw_data_dict = self.get_mfcc_samples_dict()
        
            # loop through all subjects to get samples
            for key, value in raw_data_dict.items():
                
                # save subject label in the mapping
                subject_label = 'Subject ' + str(key)
                print("\nProcessing: {}".format(subject_label))
                data["mapping"].append(subject_label)       # Subject label
                data["session_lengths"].append(value[1])   # List[subject][session_length_list]

                # process all samples per subject
                for i, sample in enumerate(value[0]):

                    data["labels"].append(key-1)    # Subject nr
                    data["mfcc"].append(sample)  # MFCC sample on same index
                     
                    print("sample:{} is done".format(i+1))
                    #print(np.array(mfcc_data).shape)

            # save MFCCs to json file
            with open(json_path, "w") as fp:
                json.dump(data, fp, indent=4)



    # OBSOLETE
    def get_reg_samples_dict(self) -> dict:
        return self.reg_samples_per_subject
    def reshape_session_df_to_signal(self, df:DataFrame):
            main_df = df[['timestamp', 1]].rename(columns={1: 'emg'})
            for i in range(2, 17):
                adding_df = df[['timestamp', i]].rename(columns={i: 'emg'})
                main_df = pd.concat([main_df, adding_df], ignore_index=True)
            samplerate = get_samplerate(main_df)
            return main_df, samplerate
    def store_samples(self, split_nr) -> None:
        for subject_nr in range(5):
            subj_samples = []
            for session_nr in range(4):
                list_of_emg = self.get_emg_list(subject_nr+1, session_nr+1)
                tot_session_df = self.make_subj_sample(list_of_emg)

                # TESTING FOR NAN
                if tot_session_df.isnull().values.any():
                    print('NaN in: subject', subject_nr+1, 'session:', session_nr+1, 'where? HERE')

                samples = np.array_split(tot_session_df.to_numpy(), split_nr)
                for array in samples:
                    df = DataFrame(array).rename(columns={0:'timestamp'})
                    df_finished, samplerate = self.reshape_session_df_to_signal(df)
                    subj_samples.append([df_finished, samplerate])
            
            self.reg_samples_per_subject[subject_nr+1] = subj_samples
    def save_json_reg(self, json_path):
        
        # Dictionary to store mapping, labels, and MFCCs
        data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
        }

        raw_data_dict = self.get_reg_samples_dict()
    
        # Loop through all subjects to get samples
        mfcc_list = []
        mfcc_frame_list = []

        for key, value in raw_data_dict.items():


            # save subject label in the mapping
            subject_label = 'Subject ' + str(key)
            data["mapping"].append(subject_label)
            print("\nProcessing: {}".format(subject_label))

            # process all samples per subject
            for i, (sample) in enumerate(value):

                # load signal from sample
                signal, sample_rate = sample[0], sample[1]
                signal = signal['emg'].to_numpy()
                test_df_for_bugs(signal, key, i)
                #print(sample_rate)

                # extract mfcc
                mfcc = mfcc_custom(signal, sample_rate, MFCC_WINDOWSIZE, MFCC_STEPSIZE, NR_COEFFICIENTS, NR_MEL_BINS)
                
                mfcc_list.append(mfcc.tolist())
                mfcc_frame_list.append(mfcc.shape[0])

                #data["mfcc"].append(mfcc.tolist())
                data["labels"].append(key-1)
                print("sample:{} is done".format(i+1))

        minimum = min(mfcc_frame_list)

        for mfcc_data in mfcc_list:

            data["mfcc"].append(mfcc_data[:minimum])
            print(np.array(mfcc_data[:minimum]).shape)

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)



# HELP FUNCTIONS: ------------------------------------------------------------------------: 

# Help: gets the str from emg nr
def get_emg_str(emg_nr):
    return 'emg' + str(emg_nr)

# Help: gets the min/max of a df
def get_min_max_timestamp(df:DataFrame):
    #min = int(np.floor(df['timestamp'].min()))
    min = df['timestamp'].min()
    max = df['timestamp'].max()
    return min, max

# Help: returns df_time_emg
def make_df_from_xandy(x, y, emg_nr):
    dict = {'timestamp': x, get_emg_str(emg_nr): y}
    df = DataFrame(dict)
    #print(df)
    return df

# Help: returns the samplerate of a df
# REMOVE 60-SECOND-BUG IF FIXED
def get_samplerate(df:DataFrame):
    min, max = get_min_max_timestamp(df)
    if max > 60 and min < 60:
        seconds = max - 60 - min
    else:
        seconds = max - min
    samples = len(df.index)
    samplerate = samples / seconds
    return int(samplerate)

# Help: takes in a df and outputs np arrays for x and y values
def get_xory_from_df(x_or_y, df:DataFrame):
    swither = {
        'x': df.iloc[:,0].to_numpy(),
        'y': df.iloc[:,1].to_numpy()
    }
    return swither.get(x_or_y, 0)

# Help: slightly modified mfcc with inputs like below. Returns N (x_values from original df) and mfcc_y_values 
def mfcc_custom(signal, samplerate, windowsize=MFCC_WINDOWSIZE, 
                                            stepsize=MFCC_STEPSIZE, 
                                            nr_coefficients=NR_COEFFICIENTS, 
                                            nr_mel_filters=NR_MEL_BINS):
        
    return mfcc(signal, samplerate, windowsize, stepsize, nr_coefficients, nr_mel_filters)

# Help: test for unregularities in DataFrame obj
def test_df_for_bugs(signal, key, placement_index):
    df = DataFrame(signal)
    if df.isnull().values.any():
        print('NaN in subject', key, 'in sample', placement_index)
    if df.shape[1] != (1):
        print('Shape:', df.shape[1], 'at subject', key, 'in sample', placement_index)



