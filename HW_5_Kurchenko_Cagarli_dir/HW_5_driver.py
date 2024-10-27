# CSCI 420
# HW 05
#
# Lindsay Cagarli
# Anna Kurchenko

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from math import log2

def calculate_entropy(labels):
    """Calculate entropy of a set of labels."""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -sum(p * log2(p) for p in probabilities if p > 0)

def information_gain_ratio(data, labels, threshold, attribute_index):
    """Calculate the Information Gain Ratio for a given attribute and threshold."""
    left_split = data[:, attribute_index] <= threshold
    right_split = ~left_split
    
    # Entropy before split
    parent_entropy = calculate_entropy(labels)

    # Entropy after split
    left_entropy = calculate_entropy(labels[left_split])
    right_entropy = calculate_entropy(labels[right_split])

    # Weighted average of child entropies
    n_left, n_right = left_split.sum(), right_split.sum()
    weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / (n_left + n_right)

    # Information gain
    gain = parent_entropy - weighted_entropy
    
    # Split information (entropy of split proportion)
    total = n_left + n_right
    split_info = -((n_left / total) * log2(n_left / total) + (n_right / total) * log2(n_right / total)) if n_left > 0 and n_right > 0 else 1
    gain_ratio = gain / split_info if split_info != 0 else 0
    
    return gain_ratio, left_split, right_split

'''
classifier result params: 
    which_attribute 
        The index of the attribute identified for the split 

    which_threshold
        Chosen threshold for data to be classified against

    which_direction
        Direction our split is going, <= or > 

    best_gain_ratio: 1.0
        Split chosen by information gain ratio, best 1 worst 0 

    left_split 
        Indicates which data points fall into left split of data after applying the condition attribute <= threshold 
    right_split
        Same but for right side 
'''
def train_decision_tree(data, labels, max_depth=8, min_leaf_size=5, current_depth=0):
     #Base case checks
    if current_depth >= max_depth:
        return None
    if len(labels) < min_leaf_size:
        return None
    if (labels == labels[0]).mean() >= 0.9:
        return None

    # Initialize best classifier for this recursion
    best_classifier = {
        'which_attribute': -1,
        'which_threshold': -1,
        'which_direction': '<=',
        'best_gain_ratio': -1,
        'left_split': None,
        'right_split': None
    }

    # Iterate over each attribute to find the best split by Information Gain Ratio
    for attribute_index in range(data.shape[1]):
        values = np.unique(data[:, attribute_index])

        for threshold in values:
            gain_ratio, left_split, right_split = information_gain_ratio(data, labels, threshold, attribute_index)
            if gain_ratio > best_classifier['best_gain_ratio']:
                best_classifier.update({
                    'which_attribute': attribute_index,
                    'which_threshold': threshold,
                    'best_gain_ratio': gain_ratio,
                    'left_split': left_split,
                    'right_split': right_split
                })
    
    # If no valid split was found, return
    if best_classifier['best_gain_ratio'] == -1:
        return None

    # Recursive calls for left and right splits
    left_data = data[best_classifier['left_split']]
    right_data = data[best_classifier['right_split']]
    left_labels = labels[best_classifier['left_split']]
    right_labels = labels[best_classifier['right_split']]
    
    if len(left_labels) >= min_leaf_size:
        train_decision_tree(left_data, left_labels, max_depth, min_leaf_size, current_depth + 1)
    if len(right_labels) >= min_leaf_size:
        train_decision_tree(right_data, right_labels, max_depth, min_leaf_size, current_depth + 1)

    return best_classifier
        

def plot_roc_curves(data, attributes, intentions):
    plt.figure(figsize=(6, 6)) 
    
    for i, attribute in enumerate(attributes):
        vals = data.iloc[:, attribute].values
        true_labels = (intentions == 2).astype(int)  # 1 for aggressive, 0 for non-aggressive

        fpr, tpr, _ = roc_curve(true_labels, vals)
        roc_auc = auc(fpr, tpr)

        if i == 0:
            linestyle = 'solid'
        elif i == 1:
            linestyle = 'dashed'
        elif i == 2:
            linestyle = 'dotted'

        plt.plot(fpr, tpr, label=f'Attribute {attribute} (AUC = {roc_auc:.2f})', linestyle=linestyle)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Three Attributes')
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal', adjustable='box')  
    plt.show()

def evaluate_classifier(classifier, validation_data, validation_labels):
    attribute = classifier['which_attribute']
    threshold = classifier['which_threshold']
    direction = classifier['which_direction']

    predictions = (validation_data[:, attribute] <= threshold) if direction == '<=' else (validation_data[:, attribute] > threshold)
    correct_predictions = np.sum(predictions == (validation_labels == 'PULL_OVER'))  # Assuming 'PULL_OVER' as target label
    total_predictions = len(validation_labels)
    
    accuracy = correct_predictions / total_predictions
    return accuracy, predictions


def calculate_confusion_matrix(predictions, actual_labels, positive_label='PULL_OVER'):
    tp = np.sum((predictions == 1) & (actual_labels == positive_label))
    tn = np.sum((predictions == 0) & (actual_labels != positive_label))
    fp = np.sum((predictions == 1) & (actual_labels != positive_label))
    fn = np.sum((predictions == 0) & (actual_labels == positive_label))
    
    print("Confusion Matrix:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")
    return tp, fp, fn, tn


def create_classifier(filename, which_attribute, the_threshold, which_direction):
    with open(f"{filename}.py", 'w') as file:
        file.write(f'''
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
                   
def {filename}(filename_in):
    THE_IMPORTANT_ATTRIBUTE = {which_attribute}
    THE_IMPORTANT_THRESHOLD = {the_threshold}
    
    data = read_all_training_data('TestSuite_C_HasGlasses.csv') 

    n_aggressive = np.sum(data.iloc[:, THE_IMPORTANT_ATTRIBUTE] {which_direction} THE_IMPORTANT_THRESHOLD)
    n_behaving = np.sum(data.iloc[:, THE_IMPORTANT_ATTRIBUTE] {'>' if which_direction == '<=' else '<='} THE_IMPORTANT_THRESHOLD)

    actual_aggressive = (data['INTENT'] == 'PULL_OVER').sum()
    actual_behaving =  (data['INTENT'] == 'letpass').sum()
    
    correctly_classified = actual_aggressive + actual_behaving
    total_records = len(data)

    accuracy = correctly_classified / total_records
    
    print(f'n_behaving_well = {{n_behaving}}')
    print(f'n_aggressive = {{n_aggressive}}')
    print(f'Accuracy = {{accuracy:.2f}}')

if __name__ == "__main__":
    input_file = sys.argv[1]
    {filename}(input_file)
    ''')
        
 
def read_all_training_data(directory):
    data = pd.read_csv(directory)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = np.floor(data[numeric_cols])

    return data       

def main():
    file = sys.argv[1] 
    data = read_all_training_data(file) 
    labels = data['INTENT'].values
    data = data.drop(columns=['INTENT']).values
    
    best_classifier = train_decision_tree(data, labels)
    
    if best_classifier:
        print("Best classifier found:", best_classifier)
        create_classifier(
            "hw_5_output_classifier", 
            best_classifier['which_attribute'], 
            best_classifier['which_threshold'], 
            best_classifier['which_direction']
        )
    
    validation_data = read_all_training_data('Validation_Data_for_420.csv') 
    validation_labels = validation_data['INTENT'].values
    validation_data = validation_data.drop(columns=['INTENT']).values

    accuracy, predictions = evaluate_classifier(best_classifier, validation_data, validation_labels)
    print(f"Classifier Accuracy: {accuracy:.2f}")
    
    tp, fp, fn, tn = calculate_confusion_matrix(predictions, validation_labels)    

if __name__ == "__main__":
    main()