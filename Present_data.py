from Handle_emg_data import *
from Signal_prep import *

# PLOT FUNCTIONS:

# Plots DataFrame objects
def plot_df(df:DataFrame):
    lines = df.plot.line(x='timestamp')
    plt.show()

# Plots ndarrays after transformations 
def plot_arrays(N, y):
    plt.plot(N, np.abs(y))
    plt.show()


# DATA FUNCTIONS

def get_data()
    



def main():

    return None
