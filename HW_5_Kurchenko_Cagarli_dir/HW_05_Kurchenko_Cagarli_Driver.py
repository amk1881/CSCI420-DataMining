# CSCI 420
# HW 05
#
# Lindsay Cagarli
# Anna Kurchenko

import sys
import os
import pandas as pd
import numpy as np
from math import log2

class DTNode:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

def calculate_entropy(labels):
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

def information_gain_ratio(data, labels, threshold, attribute_index):
    left_mask = data[:, attribute_index] <= threshold
    right_mask = ~left_mask
    
    # If split creates empty node, return 0
    if not left_mask.any() or not right_mask.any():
        return 0, left_mask, right_mask
    
    parent_entropy = calculate_entropy(labels)
    left_entropy = calculate_entropy(labels[left_mask])
    right_entropy = calculate_entropy(labels[right_mask])
    
    #  weighted entropy
    n_left, n_right = left_mask.sum(), right_mask.sum()
    total = n_left + n_right
    weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / total
    
    info_gain = parent_entropy - weighted_entropy
    split_info = -((n_left/total) * log2(n_left/total) + (n_right/total) * log2(n_right/total))
    
    # to avoid division by zero
    gain_ratio = info_gain / split_info if split_info != 0 else 0
    
    return gain_ratio, left_mask, right_mask


def get_majority_class(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

'''
Train a DT recursively using the DecisionTree Node class 
'''
def train_dt(data, labels, feature_names, max_depth=8, min_samples=5, current_depth=0):
    # Base cases
    if current_depth >= max_depth or len(labels) < min_samples:
        return DTNode(label=get_majority_class(labels))
    
    # Check if the node is pure enough (90% one class)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 1 or max(counts) / len(labels) >= 0.9:
        return DTNode(label=get_majority_class(labels))
    
    best_gain = -1
    best_split = None
    
    for feature_idx in range(data.shape[1]):
        unique_values = np.unique(data[:, feature_idx])
        for threshold in unique_values:
            gain, left_mask, right_mask = information_gain_ratio(data, labels, threshold, feature_idx)
            
            if gain > best_gain:
                best_gain = gain
                best_split = {
                    'feature_idx': feature_idx,
                    'threshold': threshold,
                    'left_mask': left_mask,
                    'right_mask': right_mask
                }
    
    # return leaf node if no good split found 
    if best_gain <= 0:
        return DTNode(label=get_majority_class(labels))
    
    left_data = data[best_split['left_mask']]
    right_data = data[best_split['right_mask']]
    left_labels = labels[best_split['left_mask']]
    right_labels = labels[best_split['right_mask']]
    
    node = DTNode(
        attribute=feature_names[best_split['feature_idx']],
        threshold=best_split['threshold'],
        left=train_dt(left_data, left_labels, feature_names, max_depth, min_samples, current_depth + 1),
        right=train_dt(right_data, right_labels, feature_names, max_depth, min_samples, current_depth + 1)
    )
    
    return node

def code_gen_helper(node, indent=""):
    if node.label is not None:
        return f"{indent}return '{node.label}'"
    
    return f"""{indent}if record['{node.attribute}'] <= {node.threshold}:
{code_gen_helper(node.left, indent + "    ")}
{indent}else:
{code_gen_helper(node.right, indent + "    ")}"""

def create_classifier_file(tree, output_filename):
    code = f"""
# CSCI 420 HW 05
# Lindsay Cagarli
# Anna Kurchenko

import pandas as pd
import numpy as np

def classify_record(record):
{code_gen_helper(tree, "    ")}

def main(filename):
    # read in data
    data = pd.read_csv(filename)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(np.floor)
    
    # Classify each record
    predictions = []
    for _, record in data.iterrows():
        prediction = classify_record(record)
        predictions.append(prediction)
    
    # Save predictions and output 
    predictions_string = ','.join(predictions) 
    with open('HW_05_Kurchenko_Cagarli_MyClassifications.csv', 'w') as f:
        f.write(predictions_string) 

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python {output_filename}.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
"""
    
    with open(f"{output_filename}.py", 'w') as f:
        f.write(code)


# prediction for single reord in DT
def predict_single(tree, record):
    if tree.label is not None:
        return tree.label
    
    if record[tree.attribute] <= tree.threshold:
        return predict_single(tree.left, record)
    
    else:
        return predict_single(tree.right, record)


def evaluate_classifier(tree, data, labels):
    predictions = []
    for _, record in data.iterrows():
        prediction = predict_single(tree, record)
        predictions.append(prediction)
    
    correct = sum(1 for true, pred in zip(labels, predictions) if true == pred)
    accuracy =  correct / len(labels)

    return accuracy, predictions


def main():
    training_data = pd.read_csv(sys.argv[1])
    
    # Balance the dataset
    aggressive = training_data[training_data['INTENT'] == 'PULL_OVER']
    non_aggressive = training_data[training_data['INTENT'] == 'letpass']
    min_samples = min(len(aggressive), len(non_aggressive))

    balanced_data = pd.concat([
        aggressive.sample(min_samples, random_state=42),
        non_aggressive.sample(min_samples, random_state=42)
    ])
    
    feature_names = balanced_data.columns.drop('INTENT').tolist()
    X = balanced_data[feature_names].values
    y = balanced_data['INTENT'].values
    
    X = np.floor(X)
    
    tree = train_dt(X, y, feature_names, max_depth=8, min_samples=5)
    
    train_accuracy, predictions = evaluate_classifier(tree, balanced_data[feature_names], balanced_data['INTENT'])
    print('training_accuracy is: ', train_accuracy)

    create_classifier_file(tree, "HW_05_Kurchenko_Cagarli_Classifier")

    important_attrs = get_important_attributes(tree)
    sorted_importance = sorted(important_attrs.items(), key=lambda item: item[1])

    print("Most Important Attributes:")
    for attribute, depth in sorted_importance:
        print(f"{attribute} (Depth: {depth})")


def get_important_attributes(node, depth=0, max_depth=3, important_attributes=None):
    if important_attributes is None:
        important_attributes = {}

    if depth >= max_depth or node is None:
        return important_attributes

    if node.attribute:
        if node.attribute not in important_attributes:
            important_attributes[node.attribute] = depth
        else:
            important_attributes[node.attribute] = min(important_attributes[node.attribute], depth)

    get_important_attributes(node.left, depth + 1, max_depth, important_attributes)
    get_important_attributes(node.right, depth + 1, max_depth, important_attributes)

    return important_attributes



if __name__ == "__main__":
    main()
