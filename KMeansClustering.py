import math
import random
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()

# Calculate Euclidean distance between two data points
def euclidean_distance(point1, point2):
    sum_squared_distance = 0.0
    for i in range(len(point1)):
        sum_squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(sum_squared_distance)

# K-Means algorithm from scratch
def k_means(data, k, max_iterations=100):
    # Step 1: Initialization
    # Randomly select 'k' unique points from the dataset to be the initial centroids
    centroids = random.sample(data, k)
    
    for iteration in range(max_iterations):
        # Create empty lists to hold the points assigned to each cluster
        clusters = [[] for _ in range(k)]
        
        # Step 2: Assignment
        # Find the closest centroid for every data point
        for point in data:
            min_dist = float('inf')
            closest_index = -1
            
            for i in range(k):
                dist = euclidean_distance(point, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
                    
            clusters[closest_index].append(point)
            
        # Store current centroids to check if they move during the update
        old_centroids = [list(c) for c in centroids]
        
        # Step 3: Update
        # Calculate the new mean for each cluster to become the new centroid
        for i in range(k):
            if not clusters[i]: # Edge case: if a cluster ends up empty, skip it
                continue
                
            new_centroid = []
            features = len(data[0])
            
            # Calculate the average for each feature (column)
            for j in range(features):
                feature_sum = sum(p[j] for p in clusters[i])
                new_centroid.append(feature_sum / len(clusters[i]))
                
            centroids[i] = new_centroid
            
        # Step 4: Convergence check
        # If the centroids are exactly the same as the last iteration, stop early
        if centroids == old_centroids:
            print(f"Algorithm converged at iteration {iteration}")
            break
            
    return centroids, clusters

# Main program
k_value = 3  # We know Iris has 3 species
final_centroids, final_clusters = k_means(X, k_value)

print("\nFinal Centroid Coordinates:")
for i, centroid in enumerate(final_centroids):
    rounded_centroid = [round(val, 3) for val in centroid]
    print(f"Cluster {i} Center: {rounded_centroid} (Contains {len(final_clusters[i])} points)")