import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def silhouette_score(data, labels):
   
    n = len(data)
    
    # Map cluster IDs to point indices
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    
    unique_labels = list(clusters.keys())
    
    # Silhouette score is undefined for fewer than 2 clusters or N clusters of size 1
    if len(unique_labels) < 2 or len(unique_labels) == n:
        raise ValueError("Silhouette score requires between 2 and N-1 clusters.")
    
    silhouette_coefficients = []
    
    for i in range(n):
        current_label = labels[i]
        same_cluster_indices = [idx for idx in clusters[current_label] if idx != i]
        
        if len(same_cluster_indices) == 0:
            a_i = 0.0
        else:
            a_i = sum(
                euclidean_distance(data[i], data[j]) for j in same_cluster_indices
            ) / len(same_cluster_indices)
        
        b_i = float('inf')
        for other_label in unique_labels:
            if other_label == current_label:
                continue
            
            other_indices = clusters[other_label]
            mean_dist_to_other_cluster = sum(
                euclidean_distance(data[i], data[j]) for j in other_indices
            ) / len(other_indices)
            
            if mean_dist_to_other_cluster < b_i:
                b_i = mean_dist_to_other_cluster
        
        if len(same_cluster_indices) == 0:
            s_i = 0.0
        else:
            max_ab = max(a_i, b_i)
            s_i = (b_i - a_i) / max_ab if max_ab > 0 else 0.0
            
        silhouette_coefficients.append(s_i)
    
    return sum(silhouette_coefficients) / n

dataset = [
    [1.0, 2.0], [1.1, 2.1], [0.9, 1.9],  
    [8.0, 8.0], [8.1, 8.2], [7.9, 8.0]  
]
cluster_labels = [0, 0, 0, 1, 1, 1]

score = silhouette_score(dataset, cluster_labels)
print(f"Overall Silhouette Index: {score:.4f}")