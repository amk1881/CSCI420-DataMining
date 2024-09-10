import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# Sources: Had to look at stack overflow for how to get Gaussian dist, 
# then again for how to plot stuff. Based on an example. 


def partA(ax):
    # Allocate 10,000 Gaussian random values, with zero mean, and a standard deviation of 1.0, allocate five vectors
    n = 10000
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)
    S = np.random.normal(0, 1, n)
    T = np.random.normal(0, 1, n)

    # a. For all of the X values, let dist = sqrt( x^2 )
        # Find the fraction of the data that is within 1 standard deviation of the origin.
    x_dist = np.sqrt(X**2)      # this is the same as taking abs(x)
    x_fraction = np.sum(x_dist <= 1.0) / n

    # For all of the (X,Y) values let dist = sqrt( x^2 + y^2 )
    xy_dist = np.sqrt(X**2 + Y**2)
    xy_fraction = np.sum(xy_dist <= 1.0) / n

    # on (X,Y, Z)
    xyz_dist = np.sqrt(X**2 + Y**2 + Z**2)
    xyz_fraction = np.sum(xyz_dist <= 1.0) / n

    # on(X, Y, Z, S)
    xyzs_dist = np.sqrt(X**2 + Y**2 + Z**2 + S**2)
    xyzs_fraction = np.sum(xyzs_dist <= 1.0) / n

    # on (X, Y, Z, S, T)
    xyzst_dist = np.sqrt(X**2 + Y**2 + Z**2 + S**2 + T**2)
    xyzst_fraction = np.sum(xyzst_dist <= 1.0) / n

    all_fractions = [x_fraction, xy_fraction, xyz_fraction, xyzs_fraction, xyzst_fraction]


    # plot data 
    ax.plot([1, 2, 3, 4, 5], all_fractions, marker='o', label='Fraction Within 1 Std Dev')
    ax.set_xlabel('Number of Elements in the Vector')
    ax.set_ylabel('Fraction Within 1 Standard Deviation')
    ax.grid(True)
    ax.legend()
    ax.set_title('Gaussian Distances')

    # Summarize fractional deviations 
    print("Part A")
    print("Fractions of data within 1 standard deviation:")
    for i, fraction in enumerate(all_fractions, start=1):
        print(f"For {i} elements: {fraction * 100:.2f}%")


# Reads and quantizes traffic data, implements Otsu's method
def partB(ax):
    data = get_data('HW01_Kurchenko_Anna_dir/Traffic_Stations')
    data['SPEED'] = data['SPEED'].round() # to nearest mile

    mixed_variances = otsu(data)
    thresholds, variances = zip(*mixed_variances)

    # Plot mixed variance vs. threshold
    ax.plot(thresholds, variances, marker='o')
    ax.set_xlabel('Threshold Speed (mph)')
    ax.set_ylabel('Mixed Variance')
    ax.grid(True)
    ax.set_title('Mixed Variance vs. Threshold Speed')



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


# for each threshhold finds false positives/negatives and plots 
def partC(ax):
    data_directory = 'HW01_Kurchenko_Anna_dir/Traffic_Stations'
    data = get_data(data_directory)
    
    thresholds = sorted(list(data['SPEED'].unique()))
    false_alarms = []
    misses = []
    mistakes = []

    total_data_count = len(data)

    for thresh in thresholds:
        # false = speeders > thresh 
        # misses = speeders <= thresh 
        false_alarms_data = data[((data['SPEED']) <= thresh) & (data['SPEED'] < 55) ]
        false_alarms_ct = len(false_alarms_data)

        misses_data = data[((data['SPEED']) > thresh) & (data['SPEED'] > 55) ]
        misses_ct = len(misses_data)

        total_mistakes = false_alarms_ct + misses_ct

        false_alarms_fraction = false_alarms_ct / total_data_count
        misses_fraction = misses_ct / total_data_count 
        mistakes_fraction = total_mistakes / total_data_count

        false_alarms.append(false_alarms_fraction)
        misses.append(misses_fraction)
        mistakes.append(mistakes_fraction)

    # Plotting
    ax.plot(thresholds, false_alarms, 'r-', marker='o', label='False Alarms Fraction')
    ax.plot(thresholds, misses, 'b-', marker='o', label='Misses Fraction')
    ax.plot(thresholds, mistakes, 'm-', marker='o', label='Mistakes Fraction')
    
    ax.set_xlabel('Threshold Speed (mph)')
    ax.set_ylabel('Fraction')
    ax.set_title('False Alarms, Misses, and Mistakes')
    ax.grid(True)
    ax.legend()


# Graph for total speeds - used for writeup
def plot_speed_histogram(data_directory):
    data = get_data(data_directory)
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



def main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create 1 row, 3 columns of subplots

    # Run each part and plot on the respective subplot
    partA(axs[0])
    partB(axs[1])
    partC(axs[2])

    plt.tight_layout()
    plt.show()

    #data_directory = 'HW01_Kurchenko_Anna_dir/Traffic_Stations'
    #plot_speed_histogram(data_directory)

if __name__ == "__main__":
    main()