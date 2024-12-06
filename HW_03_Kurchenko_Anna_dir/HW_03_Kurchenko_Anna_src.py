import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


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



# for each threshhold finds false positives/negatives
# finds badness for each speed and returns best threshold
# that minimizes badness
# this is #4 
def getMinimizedBadness(data):
    thresholds = sorted(list(data['SPEED']))
    total_badness = {}

    min_badness = float('inf')   
    best_thresh = None

    for thresh in thresholds:
        # FA: false alarms/false pos= speeders > thresh 
        # FN: misses/ false negatives = speeders <= thresh 
        # TP: True pos = 1- FN
        false_alarms_data = data[((data['SPEED']) <= thresh) & (data['SPEED'] < 55) ]
        false_alarms_ct = len(false_alarms_data)

        #FA = data[data['INTENT'] <= 1]['SPEED' > thresh].values 
        #FN = data[data['INTENT'] == 2]['SPEED' <= thresh].values
        #TP = data[data['INTENT'] == 2]['SPEED' > thresh].values
        FA = data[(data['INTENT'] <= 1) & (data['SPEED'] > thresh)].values
        FN = data[(data['INTENT'] == 2) & (data['SPEED'] <= thresh)].values
        TP = data[(data['INTENT'] == 2) & (data['SPEED'] > thresh)].values
       
        FA_ct = len(FA)
        FN_ct = len(FN)
        TP_ct = len(TP)

        badness= (0.3) * FA_ct  + (0.6) *FN_ct  + 0.1 * TP_ct

        total_badness[thresh] = badness

        if badness < min_badness:
            best_thresh = thresh
            min_badness = badness

    return best_thresh



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


def write_classifier(output_file, threshold):
    with open(output_file, 'w') as f:
        f.write(f'''
import csv
import numpy as np
import pandas as pd
import sys

                
def classify_cars(speeds, threshold):
    below_thresh = sum(speed <= threshold for speed in speeds)
    above_thresh = sum(speed > threshold for speed in speeds)
    return below_thresh, above_thresh

def main(input_file):
    data = pd.read_csv(input_file)
    data['SPEED'] = np.floor(data['SPEED']) # truncate data
    speeds = sorted(list(data['SPEED']))
    below_thresh, above_thresh = classify_cars(speeds, {threshold})

    print("Cars with speed <= {threshold} : ", below_thresh)
    print("Cars with speed > {threshold} : ", above_thresh)

if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
''')



'''
Finds the best threshold with minimized badness from archive
dataset
Then writes classifier to do same thing with test data. 
'''
def main():
    data = get_data('HW_03_Kurchenko_Anna_dir/Data_Archive/')
    data['SPEED'] = np.floor(data['SPEED']) # truncate data
    thresholds = sorted(list(data['SPEED']))
    #plot_speed_histogram(data)

    best_speed = getMinimizedBadness(data)
    print(best_speed)

    write_classifier("HW_03_Kurchenko_Anna_Classifier.py", best_speed)



if __name__ == "__main__":
    main() 