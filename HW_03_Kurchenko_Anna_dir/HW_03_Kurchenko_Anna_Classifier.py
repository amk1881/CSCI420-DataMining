
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
    below_thresh, above_thresh = classify_cars(speeds, 62)

    print("Cars with speed <= 62 : ", below_thresh)
    print("Cars with speed > 62 : ", above_thresh)

if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
