import pandas as pd
from pathlib import Path

class CSV_handler:

    def __init__(self):
        self.working_dir = str(Path.cwd())

    # Makes dataframe from the csv files in the working directory
    def make_df(self, filename):
        filepath = self.working_dir + str(filename)
        df = pd.read_csv(filepath)
        return df

    # Extracts out the timestamp and the selected emg signal into a new dataframe 
    def get_time_emg_table(self, filename, emg_nr):

        tot_data_frame = self.make_df(self, filename)
        emg_str = "emg" + str(emg_nr)
        filtered_df = tot_data_frame[["timestamp", emg_str]]
        return filtered_df

    




