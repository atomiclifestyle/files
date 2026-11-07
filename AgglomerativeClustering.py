import math

def euclidean_distance(point1, point2):
    """Computes the Euclidean distance between two n-dimensional points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def compute_cluster_distance(cluster_a, cluster_b, linkage='single'):
    """Calculates distance between two clusters based on the specified linkage rule."""
    pair_distances = [
        euclidean_distance(p1, p2) 
        for p1 in cluster_a 
        for p2 in cluster_b
    ]
    
    if linkage == 'single':
        return min(pair_distances)
    elif linkage == 'complete':
        return max(pair_distances)
    elif linkage == 'average':
        return sum(pair_distances) / len(pair_distances)
    else:
        raise ValueError(f"Unknown linkage option: {linkage}")

def agglomerative_clustering(data, n_clusters=2, linkage='single'):
    """
    Pure Python implementation of Agglomerative Hierarchical Clustering.
    
    :param data: List of data points, where each point is a list/tuple of numbers.
    :param n_clusters: Desired final number of clusters.
    :param linkage: Linkage criterion ('single', 'complete', or 'average').
    :return: List of clusters (each cluster is a list of points).
    """
    clusters = [[point] for point in data]
    
    while len(clusters) > n_clusters:
        min_dist = float('inf')
        to_merge = (None, None)
        
        num_clusters = len(clusters)
        
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                dist = compute_cluster_distance(clusters[i], clusters[j], linkage=linkage)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        
        i, j = to_merge
        clusters[i].extend(clusters[j])
        clusters.pop(j)
        
    return clusters


dataset = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
]

single_results = agglomerative_clustering(dataset, n_clusters=2, linkage='single')
print("Single Linkage Clusters:")
for idx, cluster in enumerate(single_results, 1):
    print(f" Cluster {idx}: {cluster}")

print("\n" + "="*40 + "\n")

complete_results = agglomerative_clustering(dataset, n_clusters=2, linkage='complete')
print("Complete Linkage Clusters:")
for idx, cluster in enumerate(complete_results, 1):
    print(f" Cluster {idx}: {cluster}")