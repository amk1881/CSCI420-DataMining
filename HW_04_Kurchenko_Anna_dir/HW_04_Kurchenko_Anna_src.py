import os
import pandas as pd
import numpy as np

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


def main():
    print("Warning: This code assumes that the attributes are all integer values and takes advantage of that.")

    data = read_all_training_data('HW_04_Kurchenko_Anna_dir/HW_05_Data_and_Test_Suite*/*.csv')

    # Initialize to sentinel values
    best_classifier = {
        'which_attribute': -1,
        'which_threshold': -1,
        'which_direction': '<=',  # Or '>', depending on how the nodes come out
        'lowest_mistakes': float('inf')
    }

    all_intentions = data['INTENT'].values # target variable
    
    for attributes in range(data.shape[1] - 1):
        attr = data.iloc[:, attributes]

        values = attr.values
        min_attribute = np.min(values)
        max_attribute = np.max(values)

        # Do the noise removal
        values = np.floor(values)

        # For each possible threshold speed, from low to high
        for threshold in range(int(min_attribute), int(max_attribute) + 1):
            leq_threshold = values <= threshold

            leq_intentions = all_intentions[leq_threshold]
            gt_intentions = all_intentions[~leq_threshold]

            n_aggressive_leq_threshold = np.sum(leq_intentions == 2)
            n_behaved_leq_threshold = np.sum(leq_intentions != 2)

            n_aggressive_gt_threshold = np.sum(gt_intentions == 2)
            n_behaved_gt_threshold = np.sum(gt_intentions != 2)

            if n_aggressive_leq_threshold > n_aggressive_gt_threshold:
                # If <= threshold, you are aggressive
                n_mistakes = n_behaved_leq_threshold + n_aggressive_gt_threshold
                direction_of_classifier = -1
            else:
                # If > threshold, you are aggressive
                n_mistakes = n_behaved_gt_threshold + n_aggressive_leq_threshold
                direction_of_classifier = 1

            if n_mistakes < best_classifier['lowest_mistakes']:
                best_classifier['which_attribute'] = attributes
                best_classifier['which_threshold'] = threshold
                best_classifier['direction'] = direction_of_classifier
                best_classifier['lowest_mistakes'] = n_mistakes

    create_classifier("HW_04_Kurchenko_Anna_Classifier", 
                      best_classifier['which_attribute'], 
                      best_classifier['which_threshold'], 
                      best_classifier['direction'])
    

def create_classifier(filename, which_attribute, the_threshold, which_direction):
    with open(f"{filename}.py", 'w') as file:
        file.write(f'''
import pandas as pd
import sys
import numpy as np
def {filename}(filename_in):
    THE_IMPORTANT_ATTRIBUTE = {which_attribute}
    THE_IMPORTANT_THRESHOLD = {the_threshold}
    all_data = pd.read_csv(filename_in)
    the_data = np.floor(all_data)

    # Aggressive drivers are > the threshold
    n_aggressive = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] > THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] <= THE_IMPORTANT_THRESHOLD)
    
    # Aggressive drivers are <= the threshold
    n_aggressive = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] <= THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(the_data.iloc[:, THE_IMPORTANT_ATTRIBUTE] > THE_IMPORTANT_THRESHOLD)

    print(f'n_behaving_well = {{n_behaving}}')
    print(f'n_aggressive = {{n_aggressive}}')


if __name__ == "__main__":
    input_file = sys.argv[1]
    {filename}(input_file)
                
                   ''')
        



if __name__ == "__main__":
    main()