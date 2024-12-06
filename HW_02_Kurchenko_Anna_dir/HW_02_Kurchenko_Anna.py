import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from HW_02_Kurchenko_Anna_Classifier import classify_driver


# Sources: Had to look at stack overflow for how to plot stuff. 
# Based on an example. 
# Otsu's method, I think I was doing this part wrong, so I also looked up how to do this online to confirm


# Traverses all directories of data and conglomerates all data 
# single data frame 
def get_data(directory):
    all_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


# Answers #4 and linearly searches each speed for FAR, TPR
# also calculates the min-mistakes 
def num4(data):
    thresholds = sorted(list(data['SPEED']))
 
    non_aggressive = data[data['INTENT'] <= 1]['SPEED'].values
    aggressive = data[data['INTENT'] == 2]['SPEED'].values

    total_non_aggressive = len(non_aggressive)
    total_aggressive = len(aggressive)

    false_alarm_rates = []
    true_pos_rates = []

    best_threshold = None
    min_mistakes = float('inf')

    for threshold in thresholds:
        # false alarms: non-aggressive above the threshold
        # true positives: aggressive above the threshold

        false_alarm_count = np.sum(non_aggressive > threshold)
        false_alarm_rate = false_alarm_count / total_non_aggressive 
        
        true_positive_count = np.sum(aggressive > threshold)
        true_pos_rate = true_positive_count / total_aggressive

        false_alarm_rates.append(false_alarm_rate)
        true_pos_rates.append(true_pos_rate)
        
        mistakes = false_alarm_count + (total_aggressive - true_positive_count)
        
        # find lowerdbound mistakes 
        if mistakes < min_mistakes:
            min_mistakes = mistakes
            best_threshold = threshold

    print(f"Best threshold found: {best_threshold}")
    print(f"Min mistakes found : {min_mistakes}")


    return best_threshold, false_alarm_rates, true_pos_rates


# bar Graph for total speeds - used for writeup
def plot_speed_histogram(data):
    data['SPEED'] = data['SPEED'].round()
    
    plt.figure(figsize=(12, 6))
    plt.hist(data['SPEED'], bins=range(int(data['SPEED'].min()), int(data['SPEED'].max()) + 1), alpha=0.6, label='All Speeds', color='gray')

    speed_limit = 55
    plt.hist(data[data['SPEED'] > speed_limit]['SPEED'], bins=range(int(data['SPEED'].min()), int(data['SPEED'].max()) + 1), alpha=0.6, label='Speeders (>55 mph)', color='red')
    plt.hist(data[data['SPEED'] <= speed_limit]['SPEED'], bins=range(int(data['SPEED'].min()), int(data['SPEED'].max()) + 1), alpha=0.6, label='Non-Speeders (≤55 mph)', color='blue')

    plt.xlabel('Speed (mph)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Speeds by Intention')
    plt.legend()
    plt.grid(True)
    plt.show()


# Compute Otsu method 
def otsu(data):
    speeds = sorted(data['SPEED'].unique())  # Unique threshold speeds
    mixed_variances = []

    for thresh in speeds:
        left = data[data['SPEED'] <= thresh]
        right = data[data['SPEED'] > thresh]

        left_fraction = len(left) / len(data)
        left_variance = left['SPEED'].var() if len(left) > 0 else 0

        right_fraction = len(right) / len(data)
        right_variance = right['SPEED'].var() if len(right) > 0 else 0

        mixed_variance = left_fraction * left_variance + right_fraction * right_variance
        mixed_variances.append((thresh, mixed_variance))

    return mixed_variances

# call question dealing with #4 
# does #6 which plots the roc curve comparing TPR, FAR, mixed var
# does #5 which calls the classifier along with some example speeds 
    #wasn't sure if for this question we should pass in all the speeds we have
def main():

    data = get_data('HW_02_Kurchenko_Anna_dir/Traffic_Stations_data')
    data['SPEED'] = np.floor(data['SPEED']) # truncate data
    thresholds = sorted(list(data['SPEED']))
    #plot_speed_histogram(data)

    #gives the smallest mixed variance in the data 
    mixed_variances = np.argmin(otsu(data))

    best_threshold, false_alarm_rates, true_pos_rates = num4(data)

    # ideal point = closest to FAR = 0 and TPR = 1
    ideal_point = np.argmin(np.sqrt(np.square(np.array(false_alarm_rates)) + np.square(1 - np.array(true_pos_rates))))
    
    best_threshold_idx = false_alarm_rates.index(np.min(false_alarm_rates))  
    #min_mixed_variance_idx = thresholds.index(mixed_variances)

    #  ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(false_alarm_rates, true_pos_rates, marker='o', linestyle='-', color='b', label="ROC Curve")
    plt.xlabel('False Alarm Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Classifier')
    plt.grid(True)

    for i, threshold in enumerate(thresholds):
        plt.annotate(f'{threshold}', (false_alarm_rates[i], true_pos_rates[i]))

    plt.scatter(false_alarm_rates[ideal_point], true_pos_rates[ideal_point], marker='s', color='magenta', s=100, label='Ideal Point')
    plt.scatter(false_alarm_rates[best_threshold_idx], true_pos_rates[best_threshold_idx], marker='s', color='red', s=100, label='Best Threshold')
    plt.scatter(false_alarm_rates[mixed_variances], true_pos_rates[mixed_variances], marker='o', edgecolor='green', facecolor='none', s=200, label='Min Mixed Variance')

    plt.legend()  # Add a legend for the markers
    plt.tight_layout()
    plt.show()

    # #5: 
    speeds_to_classify = [60, 66, 70, 55, 68]

    print("\nClassifications using the best threshold:")
    for speed in speeds_to_classify:
        intent = classify_driver(speed, best_threshold)
        print("For speed " , speed , ", inent is: ", intent)


if __name__ == "__main__":
    main() 