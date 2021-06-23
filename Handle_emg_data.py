import pandas as pd
from pathlib import Path

class Data_container:
      
    def __init__(self):
        self.subject_nr
        self.subject_name
        self.data_dict
    
    

class CSV_handler:

    def __init__(self):
        self.working_dir = str(Path.cwd())
        self.data_container_dict = {i: []}

    # Makes dataframe from the csv files in the working directory
    def make_df(self, filename):
        filepath = self.working_dir + str(filename)
        df = pd.read_csv(filepath)
        return df

    # Extracts out the timestamp and the selected emg signal into a new dataframe 
    def get_time_emg_table(self, filename: str, subject_nr: int, which_arm: str, emg_nr: int):
        tot_data_frame = self.make_df(filename)
        emg_str = 'emg' + str(emg_nr)
        filtered_df = tot_data_frame[["timestamp", emg_str]]

        #self.data_dict[subject_nr] = [which_arm, emg1]

        return filtered_df
    
    def get_emg_str(emg_nr):
        return 'emg' + str(emg_nr)

    def get_min_max_timestamp():





