# Vehicle-Routing-Optimisation-with-Capacity-Aware-K-Means-Clustering-Project

- A heterogeneous vehicle routing scenario was modelled to represent last-mile delivery operations in a large urban environment (Anatolian side of Istanbul). The system included a fleet of 32 vehicles with different capacity profiles (trucks and vans), the purpose was reducing the number of vehicles used and improving the total cost and the total distance traveled with respect to certain operational constraints.

- K-means clustering algorithm was developed and applied for ensuring non-empty clusters and operational feasibility.

- Implemented capacity-aware cluster validation and refinement mechanisms to the k-means clustering algorithm incorporating demand-capacity checks, maximum service limits per route, demand-based vehicle assignment, and controlled cluster splitting and merging.

- Clusters which were created with the k-means function were sorted from largest to smallest according to their demand values, and the clusters with the highest demand values were planned to be assigned to trucks and the remaining ones to vans.

- Clusters exceeding the vehicle capacity and maximum number of nodes per cluster were divided into sub-clusters, ensuring that they did not exceed the vehicle capacity and number of points. Established subclusters were defined as big and small according to the number of customers in the clusters, and big and small clusters were merged.

- With the adjusted K-means algorithm, the number of clusters, which was initially 32, was reduced to 28; therefore, a significant improvement was made in the number of vehicles used and the total cost.

- Formulated a Traveling Salesman Problem (TSP) mathematical model for each feasible cluster by defining sets, parameters, decision variables, objective function, and constraints to minimise total travel distance and cost.

- Established clusters with the adjusted K-means algorithm were inserted into the TSP mathematical model code to find the optimal visiting sequence for each cluster. TSP mathematical model was coded using Python, and Gurobi was used as an optimization solver tool.
