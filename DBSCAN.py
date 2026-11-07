import math

def euclidean_distance(point1, point2):
    """Computes the Euclidean distance between two n-dimensional points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def region_query(data, point_idx, eps):
    """Finds all point indices within distance `eps` of `data[point_idx]`."""
    return [i for i, point in enumerate(data) if euclidean_distance(data[point_idx], point) <= eps]

def dbscan(data, eps, min_samples):
    """
    Pure Python implementation of DBSCAN Clustering.
    
    :param data: List of data points, where each point is a list/tuple of numbers.
    :param eps: Maximum neighborhood distance.
    :param min_samples: Minimum number of points required to form a core point.
    :return: List of cluster IDs for each point (-1 indicates noise).
    """
    # Labels: None = Unvisited, -1 = Noise, >= 0 = Cluster ID
    labels = [None] * len(data)
    cluster_id = 0

    for i in range(len(data)):
        # Skip point if already processed
        if labels[i] is not None:
            continue

        # Find initial neighbors
        neighbors = region_query(data, i, eps)

        # Label as Noise if neighbors are less than min_samples threshold
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            # Core point found: start expanding a new cluster
            labels[i] = cluster_id
            
            # Queue for breadth-first traversal
            queue = list(neighbors)
            queue_set = set(queue)  # Prevents redundant duplicate processing
            
            for neighbor_idx in queue:
                # Change previous Noise point to Border point of current cluster
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id

                # If neighbor has not been visited yet
                if labels[neighbor_idx] is None:
                    labels[neighbor_idx] = cluster_id
                    
                    # Check if neighbor is also a core point
                    neighbor_neighbors = region_query(data, neighbor_idx, eps)
                    if len(neighbor_neighbors) >= min_samples:
                        for nbr in neighbor_neighbors:
                            if nbr not in queue_set:
                                queue.append(nbr)
                                queue_set.add(nbr)
            
            cluster_id += 1

    return labels


dataset = [
    [1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.2, 2.0], [1.0, 1.8],
    [8.0, 8.0], [8.1, 8.2], [7.9, 8.0], [8.0, 7.8],
    [5.0, 5.0], [15.0, 15.0]
]

eps_radius = 1.0
min_pts = 3
cluster_assignments = dbscan(dataset, eps=eps_radius, min_samples=min_pts)

print(f"{'Point':<18} | {'Cluster ID':<10}")
print("-" * 32)
for point, label in zip(dataset, cluster_assignments):
    status = f"Cluster {label}" if label != -1 else "Noise (-1)"
    print(f"{str(point):<18} | {status}")