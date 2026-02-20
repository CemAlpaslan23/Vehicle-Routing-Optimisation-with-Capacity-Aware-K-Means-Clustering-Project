import pandas as pd
import numpy as np
from math import sqrt
import gurobipy
from gurobipy import GRB, Model, quicksum
import geopy.distance
import matplotlib.pyplot as plt
import math  

# Haversine Distance calculation function    
def geodesic(lat1, lon1, lat2, lon2): 
    R = 6371  # Earth's radius in kilometers

    # convert decimal degrees to radians
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    # Apply Haversine Formula
    a = math.sin(dLat / 2)**2 + math.sin(dLon / 2)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# Reading the dataset
df = pd.read_excel("data TSP.xlsx", index_col=0)

# Creating the nodes
nodes = df.index.tolist()

xcoord = df['latitude'].tolist() # Reading X-coordinate from the dataset
ycoord = df['longitude'].tolist() # Reading Y-coordinate from the dataset

# Haversine distance calculation, Dij parameter
C_geo = [[geodesic(xcoord[i],ycoord[i],xcoord[j],ycoord[j]) for j in range(len(xcoord))] for i in range(len(xcoord))]

# Convert C_geo to a NumPy array
C_geo = np.array(C_geo)

# Defining the model
TSP = gurobipy.Model()

# # Adding the optimization parameters
# TSP.setParam('MIPGap', 0.01) # Enforcing the optimality gap
# TSP.setParam('Method', 2)  # Selecting the optimization method (1: primal simplex, 2: dual simplex, 0: automatic)
# TSP.setParam('MIPFocus', 3)  # Selecting the MIP focus level (0: balance, 1: quality, 2: focus, 3: speed)
# TSP.setParam('Cuts', 2)  # Activating the cuts (0: disabled, 1: restricted, 2: aggressive)
# TSP.setParam('Heuristics', 1)  # Adjusting the frequency of use of Heuristics (0: disabled, 1: frequent)
# TSP.setParam('TimeLimit', 3600)  # Adding time limit (second)

# Defining the decision variables
x = TSP.addVars(nodes, nodes, lb=0, ub=1, vtype=GRB.BINARY, name='X') # 1 if arc (i, j) is traversed; 0 otherwise
u = TSP.addVars(nodes, lb = 0, vtype = GRB.INTEGER, name = 'U') # Decision variable for the subtour elimination constraint

# Defining the objective function, goal is minimizing the total distance traveled
TSP.setObjective(quicksum(C_geo[nodes.index(i), nodes.index(j)] * x[i, j] for i in nodes for j in nodes if i != j), GRB.MINIMIZE)

# Each customer is visited exactly once constraints
TSP.addConstrs(quicksum(x[i, j] for j in nodes if j != i) == 1 for i in nodes)

TSP.addConstrs(quicksum(x[i, j] for i in nodes if i != j) == 1 for j in nodes)

# Subtour elimination constraint
TSP.addConstrs((u[i] - u[j] + (len(nodes) - 1) * x[i, j] <= len(nodes) - 2 for i in nodes for j in nodes if i != j and i != nodes[0] and j != nodes[0]))

# Finding the solution
TSP.update()
TSP.optimize()

status = TSP.status
object_Value = TSP.objVal

print()
print("Model status is: ", status)
print()
print("Objective Function value is: ", object_Value)
print()

# Printing the decision varibales which are not zeros
if status !=3 and status != 4:
    for v in TSP.getVars():
        if TSP.objVal < 1e+99 and v.x!=0:
            print('%s %f'%(v.Varname,v.x))

# Extracting the tour
optimal_tour = [nodes[0]]  # Starting from the depot

current_node = nodes[0]
while True:
    # Find the next node in the tour
    for j in nodes:
        if j != current_node and x[current_node, j].x == 1:
            optimal_tour.append(j)
            current_node = j
            break
    # If we return to the depot, the tour is complete
    if current_node == nodes[0]:
        break

# Print the optimal tour
print()
print("Optimal TSP Tour for the Cluster is:", " -> ".join(str(node) for node in optimal_tour))

# Print the cost for the established tour
print()
print("Total Cost for the TSP Tour is: ", 1000 + 3.761 * object_Value) # This is for van
#print("Total Cost for the TSP Tour is: ", 2000 + 6.674 * object_Value) # This is for truck

# Visualization Part
# Plotting the tour
plt.figure(figsize=(14, 10))

# Plotting the optimal tour
for i in range(len(optimal_tour) - 1):
    plt.plot([ycoord[nodes.index(optimal_tour[i])], ycoord[nodes.index(optimal_tour[i + 1])]],
             [xcoord[nodes.index(optimal_tour[i])], xcoord[nodes.index(optimal_tour[i + 1])]], 'g-', lw=1)

# Adding markers for nodes
for i, node in enumerate(nodes):
    if i == 0:
        plt.plot(ycoord[i], xcoord[i], 'rs')  # Depot
        plt.text(ycoord[i], xcoord[i], 'Depot', fontsize=10)
    else:
        plt.plot(ycoord[i], xcoord[i], 'bo')  # Customer nodes
        plt.text(ycoord[i], xcoord[i], str(node), fontsize=10)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Optimal TSP Tour for the Cluster')
plt.grid(True)
plt.show()
                               