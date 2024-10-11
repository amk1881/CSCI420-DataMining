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
import matplotlib.pyplot as plt
import seaborn as sns


output_dir = 'HW_5_Kurchenko_Cagarli_dir'

# Ensure the output directory exists 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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

    classifier_path = os.path.join(output_dir, "HW_05_Kurchenko_Cagarli_Classifier.py")
    with open(classifier_path, "w") as f:
        f.write(classifier_code)


"""
Generates scatter plots for each attribute versus every other 
    """
def plot_scatter_matrix(df):

    attributes = df.columns
    num_attributes = len(attributes)

    # ensure number of scatter plots and grid size for subplots
    num_plots = int(num_attributes * (num_attributes - 1) / 2)
    grid_size = int(np.ceil(np.sqrt(num_plots)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()  

    plot_index = 0
    for i in range(num_attributes - 1):
        for j in range(i + 1, num_attributes):
            ax = axes[plot_index]  # Get the current subplot
            ax.scatter(df[attributes[i]], df[attributes[j]], alpha=0.5)
            ax.set_title(f'{attributes[i]} vs {attributes[j]}')
            ax.set_xlabel(attributes[i])
            ax.set_ylabel(attributes[j])
            ax.grid(True)
            plot_index += 1

    # Remove unused subplots
    for ax in axes[plot_index:]:
        ax.remove()

    plt.tight_layout()

    # Save 
    plot_path = os.path.join(output_dir, "genereated_scatter_matrix.png")
    plt.savefig(plot_path)    
    plt.close()

    print("Saved scatter plots matrix as 'generated_scatter_matrix.png'")


"""
Generates a heatmap to visualize the correlation between attributes.
"""
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")

    # Save map
    heatmap_path = os.path.join(output_dir, "generated_correlation_matrix.png")
    plt.savefig(heatmap_path)
    
    print("Saved correlation matrix heatmap as 'generated_correlation_matrix.png'")
    plt.close()


#    
# Program Driver
#
def main():
    
    data = read_all_training_data('HW_5_Kurchenko_Cagarli_dir/Combined_Data_for_Easy_Analysis__v45.csv')

    # EDA on our data
    plot_scatter_matrix(data)
    plot_correlation_matrix(data)

    create_classifier()

if __name__ == '__main__':
    main()