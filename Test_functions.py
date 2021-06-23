from Handle_emg_data import CSV_handler
import matplotlib.pyplot as plt

def test_df_extraction(emg_nr):
    handler = CSV_handler()

    file = "/Exp20201205_2myo_hardTypePP/HaluskaMarek_20201207_1810/myoLeftEmg.csv"
    subject1_left_emg1 = handler.get_time_emg_table(file, emg_nr)
    print(subject1_left_emg1.head)

    return subject1_left_emg1, emg_nr



# running
handler = CSV_handler
df, emg_nr = test_df_extraction(1)
print(emg_nr)
emg_str = handler.get_emg_str(emg_nr)
lines = df.plot.line(x='timestamp', y=emg_str)
plt.show()