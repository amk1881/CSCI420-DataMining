# CSCI 420
# HW 6
#
# Anna Kurchenko
# Lindsay Cagarli
#

# NOTE: There is an extra, across the board, 
# 25% penalty in this assignment for code that 
# cannot be easily read

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import csv


FILENAME = 'HW_CLUSTERING_SHOPPING_CART_v2241a.csv'
OUTPUT_FILE = 'agglomeration_out.txt'
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
def agglomerative_clustering(data): 
    OUTPUT_FILE = 'clustering_output.txt'  
    smallest_clusters = []

    # Initialize each point as its own cluster
    cluster_sizes = [1] * len(data)  
    clusters = [{i: data[i]} for i in range(len(data))]  
    cluster_centers = list(data)  # Centers start as the points themselves

    # Track cluster assignments for each record
    record_to_cluster = {i: i for i in range(len(data))}  # Maps record to cluster ID

    with open(OUTPUT_FILE, 'w') as file:
        file.write("Cluster Merging Process:\n\n")

        # Clustering loop
        while len(clusters) > 1:
            min_distance = float('inf')
            to_merge = (0, 0)

            # Find clusters with minimum L1 (Manhattan) distance
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)): 

                    # L1 distance calculation   
                    dist = np.sum(np.abs(cluster_centers[i] - cluster_centers[j]))
                    
                    # Check for new minimum distance
                    if dist < min_distance:
                        min_distance = dist
                        to_merge = (i, j)

            # Merge clusters with indices `idx1` and `idx2`
            idx1, idx2 = to_merge
            new_cluster = {**clusters[idx1], **clusters[idx2]}  # merged cluster
            #new_center = np.mean(np.array(list(new_cluster.values())), axis=0)
            new_center = np.mean(list(new_cluster.values()))    # get new centroid

            # Update merged cluster center and structure
            cluster_centers[idx1] = new_center
            clusters[idx1] = new_cluster
            cluster_sizes[idx1] = len(new_cluster)

            file.write(f"Cluster Size: {len(new_cluster)}, Average Prototype: {new_center}\n")
            
            for record_id in new_cluster.keys():
                record_to_cluster[record_id] = idx1

            # Log size of the smallest merged cluster
            smallest_cluster_size = min(len(clusters[idx1]), len(clusters[idx2]))

            smallest_clusters.append(smallest_cluster_size)
            if len(smallest_clusters) > 20:  # Keep only last 20
                smallest_clusters.pop(0)

            # Remove second cluster data
            del clusters[idx2]
            del cluster_centers[idx2]
            del cluster_sizes[idx2]

        # Report the final merged clusters
        file.write("\nFinal Cluster Assignments:\n")
        for record_id, cluster_id in record_to_cluster.items():
            file.write(f"Record {record_id} -> Cluster {cluster_id}\n")


        final_cluster_sizes = sorted([len(cluster) for cluster in clusters])
        print("\nCluster Sizes (from smallest to largest):", final_cluster_sizes)

        final_prototypes = [center for center in cluster_centers]
        print("\nAverage Prototypes of Each Cluster:", final_prototypes)

    # Return last 10 smallest cluster sizes from the last 20 merges
    return smallest_clusters[-10:]


'''
Alternate clustering mechanism that includes a target cluster size to stop at 
'''
def agglomerative_clustering_target_size(data, target_clusters=5): 
    OUTPUT_FILE = 'clustering_output.txt'  
    smallest_clusters = []
    clusters = [{i: data[i]} for i in range(len(data))]  # Initial clusters
    cluster_centers = list(data)  # Initial cluster centers

    with open(OUTPUT_FILE, 'w') as file:
        file.write("Cluster Merging Process:\n\n")

        # Continue merging until we reach the target number of clusters
        while len(clusters) > target_clusters:
            min_distance = float('inf')
            to_merge = (0, 0)

            # Find clusters with minimum L1 (Manhattan) distance
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    dist = np.sum(np.abs(cluster_centers[i] - cluster_centers[j]))
                    if dist < min_distance:
                        min_distance = dist
                        to_merge = (i, j)

            # Merge clusters with indices `idx1` and `idx2`
            idx1, idx2 = to_merge
            new_cluster = {**clusters[idx1], **clusters[idx2]}  # Merged cluster
            new_center = np.mean(np.array(list(new_cluster.values())), axis=0)

            # Update merged cluster center and structure
            cluster_centers[idx1] = new_center
            clusters[idx1] = new_cluster

            file.write(f"Cluster Size: {len(new_cluster)}, Average Prototype: {new_center}\n")

            del clusters[idx2]
            del cluster_centers[idx2]

        # Sort final clusters by size and report
        final_cluster_sizes = sorted([len(cluster) for cluster in clusters])
        file.write("\nFinal Cluster Sizes (from smallest to largest):\n")
        file.write(", ".join(map(str, final_cluster_sizes)) + "\n")

    # Return sorted sizes of remaining clusters
    return final_cluster_sizes


'''
Creates a dendrogram based off clustered data
'''
def create_dendrogram(data):
    plt.figure(0)
    data = linkage(data[:,1:], 'centroid') 
    dendrogram(data, p=80, truncate_mode='lastp')
    plt.show()
    

# Program driver
def main():
    data = get_data()
    #print_cross_correlations(data)
    last_10_smallest_clusters = agglomerative_clustering(data)
    #final_cluster_sizes = agglomerative_clustering_target_size(data, target_clusters=5)

    print("\nLast 10 smallest clusters merged:", last_10_smallest_clusters)
    create_dendrogram(data)
 
if __name__ == "__main__":
    main()

