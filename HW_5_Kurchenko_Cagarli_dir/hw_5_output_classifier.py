
import pandas as pd
import numpy as np

def read_all_training_data(directory):
    data = pd.read_csv(directory)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = np.floor(data[numeric_cols])

    return data
                   
def hw_5_output_classifier(filename_in):
    THE_IMPORTANT_ATTRIBUTE = 2
    THE_IMPORTANT_THRESHOLD = 0.0
    
    data = read_all_training_data('TestSuite_C_HasGlasses.csv') 

    n_aggressive = np.sum(data.iloc[:, THE_IMPORTANT_ATTRIBUTE] <= THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(data.iloc[:, THE_IMPORTANT_ATTRIBUTE] > THE_IMPORTANT_THRESHOLD)

    print(f'n_behaving_well = {n_behaving}')
    print(f'n_aggressive = {n_aggressive}')

if __name__ == "__main__":
    input_file = sys.argv[1]
    hw_5_output_classifier(input_file)
                   