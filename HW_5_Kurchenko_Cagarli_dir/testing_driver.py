import os
import numpy as np
import pandas as pd
from collections import Counter

output_dir = 'HW_5_Kurchenko_Cagarli_dir'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to read and truncate training data
def read_all_training_data(directory_pattern):
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

# Gini impurity function
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        class_values = [row[-1] for row in group]
        for class_val in classes:
            p = class_values.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create terminal node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Generate classifier program
def create_classifier(tree):
    def create_node_code(node, depth):
        indent = '    ' * depth
        if isinstance(node, dict):
            code = f"{indent}if data[row][{node['index']}] < {node['value']}:\n"
            code += create_node_code(node['left'], depth + 1)
            code += f"{indent}else:\n"
            code += create_node_code(node['right'], depth + 1)
        else:
            code = f"{indent}return {node}\n"
        return code

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


def read_data(directory_pattern):
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

def classify(data):
    for row in range(len(data)):
"""
    classifier_code += create_node_code(tree, 2)
    classifier_code += f"""
# Program driver
def main():
    if len(sys.argv) > 1:
        # Load validation data
        validation_data = read_data('HW_5_Kurchenko_Cagarli_dir/'+ sys.argv[1])
       # validation_data = np.loadtxt(sys.argv[1], delimiter=',')
        for row in range(len(validation_data)):
            intent = classify(validation_data)
            print(intent)
    else:
        print("ERR: Missing Test_Suite Directory")

if __name__ == '__main__':
    main()
"""

    classifier_path = os.path.join(output_dir, "test_classifier.py")
    with open(classifier_path, "w") as f:
        f.write(classifier_code)

# Program driver
def main():
    
    data = read_all_training_data('HW_5_Kurchenko_Cagarli_dir/Combined_Data_for_Easy_Analysis__v45.csv')
    tree = build_tree(data.values.tolist(), max_depth=5, min_size=10)
    create_classifier(tree)

if __name__ == '__main__':
    main()
