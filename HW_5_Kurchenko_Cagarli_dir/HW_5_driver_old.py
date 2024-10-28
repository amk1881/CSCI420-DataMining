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


class DecisionTreeNode:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


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
def train_decision_tree2(data, labels, column_names, max_depth=8, min_leaf_size=5, current_depth=0):
     #Base case checks
    if current_depth >= max_depth:
        return None
    if len(labels) < min_leaf_size:
        return None
    if (labels == labels[0]).mean() >= 0.9:
        return None

    # Initialize best classifier for this recursion
    best_classifier = {
        'which_attribute': None,
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
                    'which_attribute': column_names[attribute_index],  # Use column name instead of index
                    'which_threshold': float(threshold),               # Convert np.float64 to standard float
                    'best_gain_ratio': float(gain_ratio),              # Convert np.float64 to standard float
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
        train_decision_tree(left_data, left_labels, column_names, max_depth, min_leaf_size, current_depth + 1)
    if len(right_labels) >= min_leaf_size:
        train_decision_tree(right_data, right_labels, column_names, max_depth, min_leaf_size, current_depth + 1)

    return best_classifier

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

def calculate_accuracy(true_labels, predictions):
    """Calculate accuracy without using sklearn."""
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    return correct / len(true_labels)

def calculate_confusion_matrix(true_labels, predictions):
    """Calculate confusion matrix without using sklearn."""
    # Get unique classes
    classes = sorted(list(set(true_labels)))
    n_classes = len(classes)
    
    # Create mapping of class labels to indices
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    
    # Initialize confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for true, pred in zip(true_labels, predictions):
        true_idx = class_to_idx[true]
        pred_idx = class_to_idx[pred]
        conf_matrix[true_idx, pred_idx] += 1
    
    return conf_matrix, classes

def predict_single(tree, record):
    """Make a prediction for a single record using the decision tree."""
    if tree.label is not None:
        return tree.label
    
    if record[tree.attribute] <= tree.threshold:
        return predict_single(tree.left, record)
    else:
        return predict_single(tree.right, record)

def predict(tree, data):
    """Make predictions for multiple records using the decision tree."""
    predictions = []
    for _, record in data.iterrows():
        prediction = predict_single(tree, record)
        predictions.append(prediction)
    return predictions

def evaluate_classifier(tree, data, labels):
    """Evaluate the classifier and print detailed metrics."""
    predictions = predict(tree, data)
    
    # Calculate accuracy manually
    accuracy = calculate_accuracy(labels, predictions)
    
    # Calculate confusion matrix manually
    conf_matrix, classes = calculate_confusion_matrix(labels, predictions)
    
    # Calculate precision, recall, and F1 score for each class
    for i, label in enumerate(classes):
        true_pos = conf_matrix[i, i]
        false_pos = conf_matrix[:, i].sum() - true_pos
        false_neg = conf_matrix[i, :].sum() - true_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics for class {label}:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
    
    print(f"\nOverall accuracy: {accuracy:.3f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, predictions

def evaluate_classifier2(classifier, validation_data, validation_labels, column_names):
    attribute = classifier['which_attribute']
    # Find the index of the attribute in the validation data
    attribute_index = column_names.get_loc(attribute)
    
    threshold = classifier['which_threshold']
    direction = classifier['which_direction']

    # Make predictions using the attribute index
    predictions = (validation_data[:, attribute_index] <= threshold) if direction == '<=' else (validation_data[:, attribute_index] > threshold)
    correct_predictions = np.sum(predictions == (validation_labels == 'PULL_OVER'))  # Assuming 'PULL_OVER' as target label
    total_predictions = len(validation_labels)
    
    accuracy = correct_predictions / total_predictions
    return accuracy, predictions



def calculate_confusion_matrix2(predictions, actual_labels, positive_label='PULL_OVER'):
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
    '''
    file = sys.argv[1] 
    data = read_all_training_data(file) 
    column_names = data.columns.drop('INTENT')  # Store column names without 'INTENT'
    labels = data['INTENT'].values
    data = data.drop(columns=['INTENT']).values
    
    best_classifier = train_decision_tree(data, labels, column_names)

# Read and preprocess training data
    if best_classifier:
    print("Best classifier found:", best_classifier)
    create_classifier(
        "hw_5_output_classifier", 
        best_classifier['which_attribute'], 
        best_classifier['which_threshold'], 
        best_classifier['which_direction']
    )
    
    validation_data = read_all_training_data('Validation_Data_for_420.csv') 
    #validation_labels = validation_data['INTENT'].values
    #validation_data = validation_data.drop(columns=['INTENT']).values

    accuracy, predictions = evaluate_classifier(best_classifier, data, labels, column_names)
    print(f"Classifier Accuracy: {accuracy:.2f}")
    
    tp, fp, fn, tn = calculate_confusion_matrix(predictions, labels)  
    '''
    training_data = pd.read_csv(sys.argv[1])

    # Balance the dataset
    aggressive = training_data[training_data['INTENT'] == 'PULL_OVER']
    non_aggressive = training_data[training_data['INTENT'] == 'letpass']
    min_samples = min(len(aggressive), len(non_aggressive))
    balanced_data = pd.concat([
        aggressive.sample(min_samples, random_state=42),
        non_aggressive.sample(min_samples, random_state=42)
    ])
    
    # Prepare features and labels
    feature_names = balanced_data.columns.drop('INTENT').tolist()
    X = balanced_data[feature_names].values
    y = balanced_data['INTENT'].values
    
    # Round numeric features
    X = np.floor(X)
    
    print("\nTraining decision tree...")
    # Train decision tree
    tree = train_decision_tree(X, y, feature_names, max_depth=8, min_samples=5)
    
    
    print("\nGenerating classifier file...")
    # Generate classifier file
    create_classifier_file(tree, "HW_05_Classifier")
    

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
    results_df.to_csv('HW_05_Cagarli_Lindsay_MyClassifications.csv', index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python {output_filename}.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
"""
    
    with open(f"{output_filename}.py", 'w') as f:
        f.write(code)

if __name__ == "__main__":
    main()