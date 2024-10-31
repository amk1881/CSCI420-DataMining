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

FILENAME = 'HW_CLUSTERING_SHOPPING_CART_v2241a.csv'
# '100_DATA_POINTS.csv'
# '2_CLUSTERING_SAMPLE.csv' 
# 'HW_CLUSTERING_SHOPPING_CART_v2241a.csv'

# all catergories given inside the data file
ATTRS = ['ID', 'FICTION', 'SCIFI', 'BABY_TODDLER', 'TEEN', 'MANGA', 'ARTHIST',
         'SELFIMPROV', 'COOKING', 'GAMES', 'GIFTS', 'JOURNALS', 'NEWS', 'NONFICT',
         'HAIRYPOTTERY', 'MYSTERIES', 'THRILLERS', 'CLASSICS', 'POETRY', 'ROMANCE',
         'HORROR']

def get_data():
    """
    Returns the data in the csv file.
    """
    return np.genfromtxt(FILENAME, delimiter=',', skip_header=1)


def print_cross_correlations(data, latex=False):
    """
    Given the data records, this function computes the cross-correlation
    coefficients of all features with all other features and output them in a
    table. It will also find the feature pairs with the highest absolute value
    coefficients and output them.
    """
    cross_correlations = np.corrcoef(data, rowvar=False)
    
    # used for formatting?
    if latex:
        print(' & '.join(map(str, [''] + ATTRS)))
        for i, row in enumerate(cross_correlations):
            normed = ['{:.2f}'.format(v) for v in row.tolist()]
            print(' & '.join(map(str, [ATTRS[i]] + normed)) + ' \\\\')
            
    else:
        print('\t'.join(map(str, [''] + ATTRS)))
        for i, row in enumerate(cross_correlations):
            normed = ['{:.2f}'.format(v) for v in row.tolist()]
            print('\t'.join(map(str, [ATTRS[i]] + normed)))
            
    # print highest correlated things bought
    print("\nTop 10 Correlations: ")
    values = {}
    for i, row in enumerate(cross_correlations):
        for j, value in enumerate(row[i + 1:]):
            if i == 0:
                continue
            values[(ATTRS[i], ATTRS[i + j + 1])] = value
            
    cross_correlations_sorted = sorted(
        values.items(), key=lambda item: abs(item[1]))[::-1]
    
    for v in cross_correlations_sorted[:10]:
        print(v)

# Agglomerative clustering function
def agglomerative_clustering(data):
    
    # Initialize each point as its own cluster and set its center
    clusters = [{i: data[i]} for i in range(len(data))]  
    cluster_centers = [data[i] for i in range(len(data))]  
    smallest_clusters = []  # List to keep sizes of smallest merged clusters

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
        new_center = np.mean(list(new_cluster.values()))  # get new centroid
        
        cluster_centers[idx1] = new_center
        clusters[idx1] = new_cluster
        
        del clusters[idx2]
        del cluster_centers[idx2]

        # Record the size of the smallest cluster being merged
        smallest_cluster_size = min(len(clusters[idx1]), len(clusters[idx2]) if idx2 < len(clusters) else 0)
        smallest_clusters.append(smallest_cluster_size)

        # Keep only the last 20 smallest cluster sizes merged
        if len(smallest_clusters) > 20:
            smallest_clusters.pop(0)

    # Return the final cluster structure and the last 10 smallest clusters merged
    return smallest_clusters[-10:]

def create_dendrogram(data):
    plt.figure(0)
    data = linkage(data[:,1:], 'centroid') 
    dendrogram(data, p=80, truncate_mode='lastp')
    plt.show()
    

def main():
    data = get_data()
    print_cross_correlations(data, latex=False)
    last_10_smallest_clusters = agglomerative_clustering(data)
    print("\nLast 10 smallest clusters merged:", last_10_smallest_clusters)
    create_dendrogram(data)

if __name__ == "__main__":
    main()
