
# CSCI 420
# HW 05
#
# Lindsay Cagarli
# Anna Kurchenko
                   
import sys
import pandas as pd
import numpy as np

def read_all_training_data(directory):
    data = pd.read_csv(directory)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = np.floor(data[numeric_cols])

    return data
                   
def hw_5_output_classifier(filename_in):
    THE_IMPORTANT_ATTRIBUTE = 12
    THE_IMPORTANT_THRESHOLD = 0.0
    
    data = read_all_training_data('TestSuite_C_HasGlasses.csv') 

    n_aggressive = np.sum(data.iloc[:, THE_IMPORTANT_ATTRIBUTE] <= THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(data.iloc[:, THE_IMPORTANT_ATTRIBUTE] > THE_IMPORTANT_THRESHOLD)

    actual_aggressive = (data['INTENT'] == 'PULL_OVER').sum()
    actual_behaving =  (data['INTENT'] == 'letpass').sum()
    
    correctly_classified = actual_aggressive + actual_behaving
    total_records = len(data)

    accuracy = correctly_classified / total_records
    
    print(f'n_behaving_well = {n_behaving}')
    print(f'n_aggressive = {n_aggressive}')
    print(f'Accuracy = {accuracy:.2f}')

if __name__ == "__main__":
    input_file = sys.argv[1]
    hw_5_output_classifier(input_file)
    