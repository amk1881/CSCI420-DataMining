#
# CSCI 420 HW05
# Classifier
#
# Lindsay Cagarli 
# Anna Kurchenko
# 

import os
import sys
import numpy as np


# This resulting decision tree classifier program looks something like this.
#    .... header and prologue ...
#    Given one parameter --
#    the string containing the filename to read in.

#    read in the input file
#    for each line in the input_test_data :
#        if ( attribute_a >= threshold_a ) :
#            if ( attribute_b > threshold_b ) :
#                intent = 2; # aggressive.
#            else :
#                intent = 1; # non-agressive.
#        else :
#            if ( attribute_c <= threshold_c ) :
#                intent = 2; # aggressive.
#            else :
#                intent = 1; # non-agressive.
#    ...
        
#    print( intent ). # print out the class value for each line of the provided validation file


def read_all_training_data(directory):
    # Get all CSV files matching the pattern
    all_csvs = [os.path.join(root, file) 
                for root, dirs, files in os.walk(directory)
                for file in files 
                if file.endswith('.csv') and 'TraffStn' in root]

    all_data = []
    for csv_file in all_csvs:
        data = pd.read_csv(csv_file)
        all_data.append(data)

    # Concatenate all data into a single dataframe and truncate values
    all_data = pd.concat(all_data, ignore_index=True)
    return np.floor(all_data)


import pandas as pd


def get_data(directory):
    all_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                all_data.append(data)
    return pd.concat(all_data, ignore_index=True)



#    
# Program Driver
# 
def main():

    print("LEN", len(sys.argv))
    if len(sys.argv) < 1:
        print("ERR: Missing Test_Suite Directory")

    print("data is: ", sys.argv[1])
    data_dir = "HW_5_Kurchenko_Cagarli_dir/" + sys.argv[1]
    print("data dir is ", data_dir)
    data = read_all_training_data(sys.argv[1])

        
    # Column names for the data
    columns = ['Speed', 'HasGlasses', 'Wears_Hat', 'Is_Eating', 'Is_Phone', 'Music_Volume', 'Is_Distracted', 
            'Car_Distance', 'Foggy_Weather', 'Lane', 'Speed_Limit', 'Turn_Signal', 'Brake_Lights', 
            'No_Lights', 'Night_Driving', 'Intent']

    data.columns = columns

    # Split features and target variable
    X = data.drop('Intent', axis=1)
    y = data['Intent']

    # Create a one-rule classifier based on the best attribute (Speed in this case)
    def classify(speed):
        if speed > 60:
            return 'letpass'
        else:
            return 'PULL_OVER'

    # Example of how to classify each record based on the 'Speed' attribute
    data['Predicted'] = data['Speed'].apply(classify)

    # Output the predictions
    for index, row in data.iterrows():
        print(f"Actual: {row['Intent']}, Predicted: {row['Predicted']}")


if __name__ == '__main__':
    main()
