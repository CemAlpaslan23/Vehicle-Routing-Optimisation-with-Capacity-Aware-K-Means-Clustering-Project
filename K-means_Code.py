import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten

# Reading the dataset
df = pd.read_excel("data.xlsx", sheet_name="customers", index_col=0)

# Number of clusters
k = 32

# Define vehicle capacities
van_capacity = 2500

truck_capacity = 11000

# Maximum points per cluster
max_points_per_cluster = 35

# Node coordinates
node_coordinates = df[['latitude', 'longitude']].values

# Reading the demand values from the dataset
demands = df['demand'].values

# Whiten the coordinates to normalize them
whitened_coordinates = whiten(node_coordinates)

# Function to perform K-means clustering ensuring non-empty clusters
def perform_kmeans(whitened_coordinates, k, max_attempts=100):
    for attempt in range(max_attempts):
        np.random.seed(attempt)  # Use attempt number as seed
        centroids, labels = kmeans2(whitened_coordinates, k, iter=100, minit='points')
        if len(np.unique(labels)) == k:  # Check if the number of unique clusters is 32
            return centroids, labels
    raise ValueError(f"K-means clustering failed to produce the required number of non-empty clusters within {max_attempts} attempts.") 



def create_clusters(centroids, labels, demands):                          
    # Initialize lists to store clusters for trucks and vans
    truck_clusters = []
    van_clusters = []                                                   
    
    # Set of all indices to keep track of unassigned points
    unassigned_indices = set(range(len(labels)))                          


    # Calculate total demand for each cluster
    cluster_demands = [(i, demands[np.where(labels == i)[0]].sum()) for i in range(k)]         
    
    # Sort clusters by total demand in descending order
    sorted_clusters = sorted(cluster_demands, key=lambda x: x[1], reverse=True)               

    
    for idx, (i, cluster_demand) in enumerate(sorted_clusters):                                                         
        # Get indices of points in the current cluster
        cluster_indices = np.where(labels == i)[0]                                             
        # Remove these indices from unassigned points
        unassigned_indices -= set(cluster_indices)                                            
        # Determine the vehicle type based on cluster index
        if idx < 2:  # First 2 clusters are assigned to trucks, which means first 2 clusters will be truck clusters                        
            capacity = truck_capacity
            vehicle_type = "Truck"
        else:
            capacity = van_capacity
            vehicle_type = "Van"

        # If cluster demand exceeds vehicle capacity or cluster has too many points (more than max_points which is 30), split it
        if cluster_demand > capacity or len(cluster_indices) > max_points_per_cluster:                                                    
            sub_cluster_indices = []
            sub_cluster_demand = 0                                                                                                        
            
            # Split cluster into sub-clusters
            for point in cluster_indices:
                # Check if adding the next point exceeds capacity or max points per cluster
                if sub_cluster_demand + demands[point] > capacity or len(sub_cluster_indices) == max_points_per_cluster:
                    if vehicle_type == "Truck":
                        truck_clusters.append((np.array(sub_cluster_indices), vehicle_type, sub_cluster_demand))
                    else:
                        van_clusters.append((np.array(sub_cluster_indices), vehicle_type, sub_cluster_demand))
                    sub_cluster_indices = []
                    sub_cluster_demand = 0                                                                                             
                                                                                                                                      
                sub_cluster_indices.append(point)
                sub_cluster_demand += demands[point]                                                                                   
            
            # Add the last sub-cluster if it exists
            if sub_cluster_indices:
                if vehicle_type == "Truck":
                    truck_clusters.append((np.array(sub_cluster_indices), vehicle_type, sub_cluster_demand))
                else:
                    van_clusters.append((np.array(sub_cluster_indices), vehicle_type, sub_cluster_demand))                          
        else:
            # Add the whole cluster if it fits the vehicle's capacity and point limit
            if vehicle_type == "Truck":
                truck_clusters.append((cluster_indices, vehicle_type, cluster_demand))
            else:
                van_clusters.append((cluster_indices, vehicle_type, cluster_demand))                                           

    # Include remaining unassigned points into new clusters
    unassigned_indices = list(unassigned_indices)
    for idx in range(0, len(unassigned_indices), max_points_per_cluster):
        selected_indices = unassigned_indices[idx:idx + max_points_per_cluster]
        van_clusters.append((selected_indices, "Van", demands[selected_indices].sum()))                                         

    # Function to merge small clusters into larger ones, smaller ones are which have less than 20 nodes 
    def merge_small_clusters(clusters, min_points):
        large_clusters = []
        small_clusters = []
        
        # Separate clusters into large and small based on min_points
        for indices, vehicle_type, total_demand in clusters:
            if len(indices) < min_points:
                small_clusters.append((indices, vehicle_type, total_demand))
            else:
                large_clusters.append((indices, vehicle_type, total_demand))                                          
        # Try to merge small clusters into existing large clusters
        for indices, vehicle_type, total_demand in small_clusters:
            merged = False
            for i in range(len(large_clusters)):
                if large_clusters[i][1] == vehicle_type and len(large_clusters[i][0]) + len(indices) <= max_points_per_cluster:
                    new_indices = np.concatenate([large_clusters[i][0], indices])
                    new_demand = large_clusters[i][2] + total_demand
                    if new_demand <= (truck_capacity if vehicle_type == "Truck" else van_capacity):
                        large_clusters[i] = (new_indices, vehicle_type, new_demand)
                        merged = True
                        break
            if not merged:
                large_clusters.append((indices, vehicle_type, total_demand))                                         

        return large_clusters                                                                                        
    # Merge small clusters for trucks and vans
    truck_clusters = merge_small_clusters(truck_clusters, 20)
    van_clusters = merge_small_clusters(van_clusters, 20)

    return truck_clusters, van_clusters                                                                             
# Perform initial k-means clustering ensuring non-empty clusters
centroids, labels = perform_kmeans(whitened_coordinates, k)

# Create clusters with specified vehicle types and capacities
truck_clusters, van_clusters = create_clusters(centroids, labels, demands)

# Adjust the customer indices by adding 1 and inserting 0 at the beginning and end
adjusted_truck_clusters = [(np.concatenate(([0], indices + 1, [0])), vehicle_type, total_demand) for indices, vehicle_type, total_demand in truck_clusters]
adjusted_van_clusters = [(np.concatenate(([0], indices + 1, [0])), vehicle_type, total_demand) for indices, vehicle_type, total_demand in van_clusters]

print("\nCluster Information with k-means Algorithm:")
print()
for idx, (indices, vehicle_type, total_demand) in enumerate(adjusted_truck_clusters + adjusted_van_clusters):
    print(f"Cluster {idx+1}: Vehicle Type: {vehicle_type}, Number of Points: {len(indices) - 2}, Total Demand: {total_demand}")
    print(f"Customers in Cluster: {indices}\n")
    
# Calculate total demand from clusters
total_demand_clusters = sum([total_demand for _, _, total_demand in adjusted_truck_clusters + adjusted_van_clusters])

# Calculate total demand from Excel data
total_demand_excel = demands.sum()

print("Total demand from clusters:", total_demand_clusters)
print("Total demand from Excel data:", total_demand_excel)

# Total number of nodes
total_nodes_excel = len(df)

# Calculate total number of nodes in clusters
total_nodes_clusters = sum([len(indices) for indices, _, _ in adjusted_truck_clusters + adjusted_van_clusters])

print("Total number of nodes from Excel data:", total_nodes_excel)
print("Total number of nodes in clusters:", total_nodes_clusters - (28 * 2)) 
print()

# Visualization Part
# Extract coordinates and cluster information
cluster_coordinates = [node_coordinates[indices - 1] for indices, _, _ in adjusted_truck_clusters + adjusted_van_clusters]
vehicle_types = ["Truck"] * len(adjusted_truck_clusters) + ["Van"] * len(adjusted_van_clusters)

# Define a colormap
cmap = plt.cm.get_cmap("tab20", k)

# Plot clusters
plt.figure(figsize=(14, 10))
for idx, (coords, vehicle_type) in enumerate(zip(cluster_coordinates, vehicle_types)):
    color = cmap(idx)
    marker = 'o' if vehicle_type == "Truck" else 's'
    plt.scatter(coords[:, 0], coords[:, 1], c=[color], label=f"Cluster {idx+1} ({vehicle_type})", marker=marker, edgecolors='k')

# Plot the depot, which is not included in the excel file
depot_latitude = 41.0113735898618
depot_longitude = 29.1755623026824
plt.scatter(depot_latitude, depot_longitude, c='red', label='Depot', marker='*', s=200, edgecolors='k')

# Adding title and labels
plt.title("Established Clusters with k-means Algorithm")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='best', bbox_to_anchor=(1, 1), ncol=1, borderaxespad=0.)
plt.grid(True)
plt.show()


