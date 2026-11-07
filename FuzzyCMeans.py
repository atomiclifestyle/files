import math
import random

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def initialize_membership_matrix(n_samples, n_clusters, seed=42):
    random.seed(seed)
    U = []
    for _ in range(n_samples):
        row = [random.random() for _ in range(n_clusters)]
        row_sum = sum(row)
        U.append([val / row_sum for val in row])
    return U

def compute_centroids(data, U, n_clusters, m):
    n_samples = len(data)
    n_features = len(data[0])
    centroids = []

    for j in range(n_clusters):
        numerator = [0.0] * n_features
        denominator = 0.0

        for i in range(n_samples):
            u_m = U[i][j] ** m
            denominator += u_m
            for f in range(n_features):
                numerator[f] += u_m * data[i][f]

        centroid = [numerator[f] / denominator for f in range(n_features)]
        centroids.append(centroid)

    return centroids

def update_membership_matrix(data, centroids, n_clusters, m):
    n_samples = len(data)
    p = 2.0 / (m - 1.0)
    new_U = []

    for i in range(n_samples):
        distances = [euclidean_distance(data[i], c) for c in centroids]

        if 0.0 in distances:
            row = [0.0] * n_clusters
            row[distances.index(0.0)] = 1.0
            new_U.append(row)
            continue

        row = []
        for j in range(n_clusters):
            d_ij = distances[j]
            sum_ratio = sum((d_ij / distances[k]) ** p for k in range(n_clusters))
            u_ij = 1.0 / sum_ratio
            row.append(u_ij)

        new_U.append(row)

    return new_U

def fuzzy_c_means(data, n_clusters=2, m=2.0, max_iter=100, tol=1e-5, seed=42):
   
    n_samples = len(data)
    U = initialize_membership_matrix(n_samples, n_clusters, seed=seed)

    for iter_num in range(max_iter):
        centroids = compute_centroids(data, U, n_clusters, m)
        new_U = update_membership_matrix(data, centroids, n_clusters, m)

        diff = max(
            abs(new_U[i][j] - U[i][j]) 
            for i in range(n_samples) 
            for j in range(n_clusters)
        )
        U = new_U

        if diff < tol:
            break

    return centroids, U

dataset = [
    [1.0, 2.0], [1.1, 2.1], [0.9, 1.9],
    [4.5, 5.0],
    [8.0, 8.0], [8.1, 8.2], [7.9, 8.0]
]

centroids, membership_matrix = fuzzy_c_means(dataset, n_clusters=2, m=2.0)

print("Computed Centroids:")
for idx, centroid in enumerate(centroids):
    print(f" Cluster {idx}: {[round(x, 4) for x in centroid]}")

print("\nMembership Matrix U (Probabilities for [Cluster 0, Cluster 1]):")
print(f"{'Data Point':<15} | {'Cluster 0':<10} | {'Cluster 1':<10} | {'Hard Label':<10}")
print("-" * 55)

for point, row in zip(dataset, membership_matrix):
    hard_label = row.index(max(row))
    c0_prob, c1_prob = row[0], row[1]
    print(f"{str(point):<15} | {c0_prob:<10.4f} | {c1_prob:<10.4f} | Cluster {hard_label}")