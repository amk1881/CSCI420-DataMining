#
# CSCI 420 HW05
# Decison Tree Trainer
#
# Lindsay Cagarli 
# Anna Kurchenko
#

import os
import numpy as np
import pandas as pd

def read_all_training_data(directory_pattern):
    # Get all CSV files matching the pattern
    all_csvs = [os.path.join(root, file) 
                for root, dirs, files in os.walk('.') 
                for file in files 
                if file.endswith('.csv') and 'TraffStn' in root]

    all_data = []
    for csv_file in all_csvs:
        data = pd.read_csv(csv_file)
        all_data.append(data)

    # Concatenate all data into a single dataframe and truncate values
    all_data = pd.concat(all_data, ignore_index=True)
    return np.floor(all_data)

#
# generate a decision tree using the data
#
def train():
   print("yay decision tree")

#
# build classifier based off 
# information gained in training
#
def create_classifier():
    
    #comparator = "<=" if direction == -1 else ">"
    
    classifier_code = f"""#
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



#    
# Program Driver
# 
def main():

    if len(sys.argv) > 1:
    
        # getting data
        print("")

    else:
        print("ERR: Missing Test_Suite Directory")

if __name__ == '__main__':
    main()
"""

    with open("HW_05_Kurchenko_Cagarli_Classifier.py", "w") as f:
        f.write(classifier_code)


#    
# Program Driver
#
def main():
    
    #data = read_all_training_data('HW_5_Kurchenko_Cagarli_dir/CHANGE_TO_DATA_FOLDER_NAME*/*.csv')
    
    create_classifier()

if __name__ == '__main__':
    main()