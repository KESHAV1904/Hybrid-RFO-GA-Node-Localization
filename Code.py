import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Step 1: Define parameters
sensor_nodes = 600  # Increased from 300 to 600
anchor_nodes = [i * 10 for i in range(1, 11)]  # Varies from 10 to 100
area_size = (300, 300)  # Deployment area 300x300 m²
transmission_range = [i * 5 for i in range(2, 9)]  # Varies from 10 to 40 meters
iterations = 100
noise_variance = [2, 6]  # Two sets of experiments
mutation_probability = 0.05  # For GA & Hybrid RFO-GA

# Initialize positions for sensor nodes randomly
def initialize_nodes(num_nodes, area):
    return np.random.uniform(0, area[0], (num_nodes, 2))

# Compute Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

# GA Localization function
def ga_localization(nodes, anchors, noise):
    localized_nodes = []
    for node in nodes:
        distances = euclidean_distance(node, anchors) + np.random.normal(0, noise, len(anchors))
        estimated_position = np.mean(anchors, axis=0)  # Placeholder: Mean estimation
        localized_nodes.append(estimated_position)
    return np.array(localized_nodes)

# RFO Localization function
def rfo_localization(nodes, anchors, noise):
    localized_nodes = []
    for node in nodes:
        distances = euclidean_distance(node, anchors) + np.random.normal(0, noise, len(anchors))
        estimated_position = np.median(anchors, axis=0)  # Placeholder: Median estimation
        localized_nodes.append(estimated_position)
    return np.array(localized_nodes)

# Hybrid RFO-GA Localization function
def hybrid_rfo_ga_localization(nodes, anchors, noise):
    localized_nodes = []
    for node in nodes:
        distances = euclidean_distance(node, anchors) + np.random.normal(0, noise, len(anchors))
        estimated_position = (np.mean(anchors, axis=0) + np.median(anchors, axis=0)) / 2  # Hybrid Estimation
        localized_nodes.append(estimated_position)
    return np.array(localized_nodes)

# Step 2: Run simulations for each algorithm
def run_algorithm(algo_name):
    target_nodes = initialize_nodes(sensor_nodes, area_size)
    anchors = initialize_nodes(random.choice(anchor_nodes), area_size)
    noise = random.choice(noise_variance)

    if algo_name == 'GA':
        localized_nodes = ga_localization(target_nodes, anchors, noise)
    elif algo_name == 'RFO':
        localized_nodes = rfo_localization(target_nodes, anchors, noise)
    elif algo_name == 'Hybrid RFO-GA':
        localized_nodes = hybrid_rfo_ga_localization(target_nodes, anchors, noise)
    else:
        raise ValueError("Invalid Algorithm Name")

    return target_nodes, anchors, localized_nodes

# Step 3: Visualization function
def plot_results(algo_name, target_nodes, anchors, localized_nodes):
    plt.figure(figsize=(8, 6))
    plt.scatter(target_nodes[:, 0], target_nodes[:, 1], c='red', label='Target Nodes', alpha=0.5)
    plt.scatter(anchors[:, 0], anchors[:, 1], c='blue', label='Anchor Nodes', marker='s')
    plt.scatter(localized_nodes[:, 0], localized_nodes[:, 1], c='green', label=f'{algo_name} Localized Nodes', alpha=0.7)
    plt.legend()
    plt.title(f'Node Localization using {algo_name}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

# Run & plot separately for GA, RFO, and Hybrid RFO-GA
for algo in ['GA', 'RFO', 'Hybrid RFO-GA']:
    target, anchors, localized = run_algorithm(algo)
    plot_results(algo, target, anchors, localized)
