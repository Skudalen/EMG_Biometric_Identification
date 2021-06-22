import pandas as pd
from pathlib import Path

class CSV_handler:

    self.working_dir

    def __init__(self):
        self.working_dir = str(Path.cwd())

    def make_df(self, filename)
        filepath = self.working_dir + str(filename)
        df = pd.read_csv(filepath)
        return df
    
    def get_time_emg_table(self, filename, emg_nr):

        tot_data_frame = make_df(self, filename)
        emg_str = "emg" + str(emg_nr)
        filtered_df = tot_data_frame[emg_str]
        return filtered_df




