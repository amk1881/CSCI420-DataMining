import sys
import os
import pandas as pd
import numpy as np
from math import log2

def calculate_entropy(labels):
    """Calculate entropy of a set of labels."""
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

def information_gain_ratio(data, labels, threshold, attribute_index):
    """Calculate the Information Gain Ratio for a given attribute and threshold."""
    left_mask = data[:, attribute_index] <= threshold
    right_mask = ~left_mask
    
    # If split creates empty node, return 0
    if not left_mask.any() or not right_mask.any():
        return 0, left_mask, right_mask
    
    parent_entropy = calculate_entropy(labels)
    left_entropy = calculate_entropy(labels[left_mask])
    right_entropy = calculate_entropy(labels[right_mask])
    
    # Calculate weighted entropy
    n_left, n_right = left_mask.sum(), right_mask.sum()
    total = n_left + n_right
    weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / total
    
    # Calculate information gain
    info_gain = parent_entropy - weighted_entropy
    
    # Calculate split information
    split_info = -((n_left/total) * log2(n_left/total) + (n_right/total) * log2(n_right/total))
    
    # Avoid division by zero
    gain_ratio = info_gain / split_info if split_info != 0 else 0
    
    return gain_ratio, left_mask, right_mask

def get_majority_class(labels):
    """Return the majority class in a set of labels."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

class DecisionTreeNode:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

def train_decision_tree(data, labels, feature_names, max_depth=8, min_samples=5, current_depth=0):
    """
    Train a decision tree recursively.
    Returns a DecisionTreeNode object representing the tree.
    """
    # Base cases
    if current_depth >= max_depth or len(labels) < min_samples:
        return DecisionTreeNode(label=get_majority_class(labels))
    
    # Check if the node is pure enough (90% one class)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 1 or max(counts) / len(labels) >= 0.9:
        return DecisionTreeNode(label=get_majority_class(labels))
    
    best_gain = -1
    best_split = None
    
    # Try each feature and threshold
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
    
    # If no good split found, return leaf node
    if best_gain <= 0:
        return DecisionTreeNode(label=get_majority_class(labels))
    
    # Create child nodes
    left_data = data[best_split['left_mask']]
    right_data = data[best_split['right_mask']]
    left_labels = labels[best_split['left_mask']]
    right_labels = labels[best_split['right_mask']]
    
    node = DecisionTreeNode(
        attribute=feature_names[best_split['feature_idx']],
        threshold=best_split['threshold'],
        left=train_decision_tree(left_data, left_labels, feature_names, max_depth, min_samples, current_depth + 1),
        right=train_decision_tree(right_data, right_labels, feature_names, max_depth, min_samples, current_depth + 1)
    )
    
    return node

def generate_python_code(node, indent=""):
    """Generate Python code for the decision tree classifier."""
    if node.label is not None:
        return f"{indent}return '{node.label}'"
    
    return f"""{indent}if record['{node.attribute}'] <= {node.threshold}:
{generate_python_code(node.left, indent + "    ")}
{indent}else:
{generate_python_code(node.right, indent + "    ")}"""

def create_classifier_file(tree, output_filename):
    """Create a Python file containing the trained classifier."""
    code = f"""
# CSCI 420 HW 05
# Lindsay Cagarli
# Anna Kurchenko

import pandas as pd
import numpy as np

def classify_record(record):
{generate_python_code(tree, "    ")}

def main(filename):
    # Read and preprocess data
    data = pd.read_csv(filename)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(np.floor)
    
    # Classify each record
    predictions = []
    for _, record in data.iterrows():
        prediction = classify_record(record)
        predictions.append(prediction)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({{'INTENT': predictions}})
    results_df.to_csv('HW_05_Kurchenko_Cagarli_MyClassifications.csv', index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python {output_filename}.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
"""
    
    with open(f"{output_filename}.py", 'w') as f:
        f.write(code)


def predict_single(tree, record):
    """Make a prediction for a single record using the decision tree."""
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
    
    tree = train_decision_tree(X, y, feature_names, max_depth=8, min_samples=5)
    
    train_accuracy, _ = evaluate_classifier(tree, balanced_data[feature_names], balanced_data['INTENT'])
    print('training_accuracy is: ', train_accuracy)
    create_classifier_file(tree, "HW_05_Classifier_Kurchenko_Cagarli")
    
    

if __name__ == "__main__":
    main()

    '''
    The improvements should significantly increase your accuracy from the current 0.23. The main reasons for the improved accuracy are:

Balanced dataset preventing bias
Better split selection using proper information gain ratio
Proper tree structure with appropriate stopping conditions
Handling of edge cases and numerical precision'''