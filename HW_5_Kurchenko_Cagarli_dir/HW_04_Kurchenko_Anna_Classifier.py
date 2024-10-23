
import pandas as pd
import sys
import numpy as np
def HW_04_Kurchenko_Anna_Classifier(filename_in):
    THE_IMPORTANT_ATTRIBUTE = 0
    THE_IMPORTANT_THRESHOLD = 69
    all_data = pd.read_csv(filename_in)
    the_data = np.floor(all_data)

    # Aggressive drivers are > the threshold
    n_aggressive = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] > THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] <= THE_IMPORTANT_THRESHOLD)
    
    # Aggressive drivers are <= the threshold
    n_aggressive = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] <= THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] > THE_IMPORTANT_THRESHOLD)

    print(f'n_behaving_well = {n_behaving}')
    print(f'n_aggressive = {n_aggressive}')


if __name__ == "__main__":
    input_file = sys.argv[1]
    HW_04_Kurchenko_Anna_Classifier(input_file)
                
                   