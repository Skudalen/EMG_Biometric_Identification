import pandas as pd
from pathlib import Path

from pandas.core.frame import DataFrame

class Data_container:
      
    def __init__(self, subject_nr:int, subject_name:str):
        self.subject_nr = subject_nr
        self.subject_name = subject_name
        self.data_dict = {'left': [None]*8, 'right': [None]*8}
    
class CSV_handler:

    def __init__(self):
        self.working_dir = str(Path.cwd()) 
        self.data_container_dict = {} # Dict with keys equal subject numbers and values equal the relvant datacontainer 

    # Makes dataframe from the csv files in the working directory
    def make_df(self, filename):
        filepath = self.working_dir + str(filename)
        df = pd.read_csv(filepath)
        return df

    # Extracts out the timestamp and the selected emg signal into a new dataframe and stores the data on the subject
    def get_time_emg_table(self, filename:str, emg_nr:int, which_arm:str, data_container:Data_container):
        tot_data_frame = self.make_df(filename)
        emg_str = 'emg' + str(emg_nr)
        filtered_df = tot_data_frame[["timestamp", emg_str]]

        # Links the retrieved data with the subjects data_container
        subject_nr = data_container.subject_nr
        self.data_container_dict[subject_nr] = data_container
        # Places the data correctly:
        if which_arm is 'left':
            data_container.data_dict['left'][emg_nr+1] = filtered_df

        return filtered_df
    
    def get_emg_str(emg_nr):
        return 'emg' + str(emg_nr)

    def get_min_max_timestamp(df:DataFrame):
        min = df['timestamp'].argmin
        max = df['timestamp'].argmax
        return min, max



