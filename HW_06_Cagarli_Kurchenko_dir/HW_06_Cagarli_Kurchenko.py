# CSCI 420
# HW 6
#
# Lindsay Cagarli
# Anna Kurchenko
#

# NOTE: There is an extra, across the board, 
# 25% penalty in this assignment for code that 
# cannot be easily read

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import csv


FILENAME = 'HW_CLUSTERING_SHOPPING_CART_v2241a.csv'
# '100_DATA_POINTS.csv'
# '2_CLUSTERING_SAMPLE.csv' 
# 'HW_CLUSTERING_SHOPPING_CART_v2241a.csv'

# all catergories given inside the data file minus ID
ATTRS = ['FICTION', 'SCIFI', 'BABY_TODDLER', 'TEEN', 'MANGA', 'ARTHIST',
         'SELFIMPROV', 'COOKING', 'GAMES', 'GIFTS', 'JOURNALS', 'NEWS', 'NONFICT',
         'HAIRYPOTTERY', 'MYSTERIES', 'THRILLERS', 'CLASSICS', 'POETRY', 'ROMANCE',
         'HORROR']

#
# returns the data in the csv file.
# and excludes first column with record id in it 
# 
def get_data():
    return np.genfromtxt(FILENAME, delimiter=',', skip_header=1)[:, 1:]

#
# finds cross correlation for all attributes and outputs to console
# and to an external file for readability
#
def print_cross_correlations(data, output_file='cross_correlations.tsv'):
    cross_correlations = np.corrcoef(data, rowvar=False)
    
    # table format results in terminal: 
    print('\t'.join(map(str, [''] + ATTRS)))
    for i, row in enumerate(cross_correlations):
        normed = ['{:.2f}'.format(v) for v in row.tolist()]
        print('\t'.join(map(str, [ATTRS[i]] + normed)))

    cross_correlations2 = np.corrcoef(data, rowvar=False)

    # Write the results to a TSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow([''] + ATTRS)  # Header row
        for i, row in enumerate(cross_correlations2):
            writer.writerow([ATTRS[i]] + ['{:.2f}'.format(v) for v in row])

    
    print("\nTop 10 Correlations: ")
    values = {}

    # for each pair of attributes
    for i in range(len(ATTRS)):
        for j in range(i + 1, len(ATTRS)):  # Only consider pairs (i, j) where j > i
            values[(ATTRS[i], ATTRS[j])] = cross_correlations[i, j]

    # sort correlations by absolute value in descending order
    cross_correlations_sorted = sorted(
        values.items(), key=lambda item: abs(item[1]), reverse=True
    )

    # print highest correlated things bought
    for pair, correlation in cross_correlations_sorted[:10]:
        print(f"{pair}: {correlation:.2f}")


#
# agglomerative clustering function
# which uses the L1 distance between points
#
def agglomerative_clustering(data):
    
    smallest_clusters = []  
    
    # initialize each point as its own cluster and set its center
    clusters = [{i: data[i]} for i in range(len(data))]  
    cluster_centers = [data[i] for i in range(len(data))]  

    # clustering loop
    while len(clusters) > 1:
        
        # find points with the min L1 distance
        min_distance = float('inf')
        to_merge = (0, 0)

        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):    
                
                # L1 distance calculation
                dist = np.sum(np.abs(cluster_centers[i] - cluster_centers[j]))
              
                # check if it is the new smallest distance
                if dist < min_distance:
                    min_distance = dist
                    to_merge = (i, j)

        # merge the two clusters
        idx1, idx2 = to_merge
        new_cluster = {**clusters[idx1], **clusters[idx2]}  # merged cluster
        new_center = np.mean(list(new_cluster.values()))    # get new centroid
        
        cluster_centers[idx1] = new_center
        clusters[idx1] = new_cluster
        
        del clusters[idx2]
        del cluster_centers[idx2]

        # record the size of the smallest cluster being merged
        smallest_cluster_size = min(len(clusters[idx1]), len(clusters[idx2]) if idx2 < len(clusters) else 0)
        smallest_clusters.append(smallest_cluster_size)

        # keep only the last 20 smallest cluster sizes merged
        if len(smallest_clusters) > 20:
            smallest_clusters.pop(0)

    # return the final cluster structure and the last 10 smallest clusters merged
    return smallest_clusters[-10:]

#
# uses external functions to create
# the dendrogram for the given data
# 
def create_dendrogram(data):
    plt.figure(0)
    data = linkage(data[:,1:], 'centroid') 
    dendrogram(data, p=80, truncate_mode='lastp')
    plt.show()
    
#
# program driver
#
def main():
    data = get_data()
    print_cross_correlations(data)
    last_10_smallest_clusters = agglomerative_clustering(data)
    print("\nLast 10 smallest clusters merged:", last_10_smallest_clusters)
    create_dendrogram(data)

if __name__ == "__main__":
    main()
