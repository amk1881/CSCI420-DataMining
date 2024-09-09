import numpy as np
import matplotlib.pyplot as plt

# Sources: Had to look at stack overflow for how to get Gaussian dist, 
# then again for how to plot stuff. Based on an example. 

def main():
    #1. Part A. 
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

    # b. For all of the (X,Y) values let dist = sqrt( x^2 + y^2 )
    xy_dist = np.sqrt(X**2 + Y**2)
    xy_fraction = np.sum(xy_dist <= 1.0) / n

    #c. on (X,Y, Z)
    xyz_dist = np.sqrt(X**2 + Y**2 + Z**2)
    xyz_fraction = np.sum(xyz_dist <= 1.0) / n

    # d. on(X, Y, Z, S)
    xyzs_dist = np.sqrt(X**2 + Y**2 + Z**2 + S**2)
    xyzs_fraction = np.sum(xyzs_dist <= 1.0) / n

    # e. on (X, Y, Z, S, T)
    xyzst_dist = np.sqrt(X**2 + Y**2 + Z**2 + S**2 + T**2)
    xyzst_fraction = np.sum(xyzst_dist <= 1.0) / n

    all_fractions = [x_fraction, xy_fraction, xyz_fraction, xyzs_fraction, xyzst_fraction]


    # plot data 
    plt.plot([1, 2, 3, 4, 5], all_fractions, marker='o', label='Fraction Within 1 Std Dev')
    plt.xlabel('Number of Elements in the Vector')
    plt.ylabel('Fraction Within 1 Standard Deviation')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Output the fractions for reference
    print("Fractions of data within 1 standard deviation:")
    for i, fraction in enumerate(all_fractions, start=1):
        print(f"For {i} elements: {fraction * 100:.2f}%")



if __name__ == "__main__":
    main()