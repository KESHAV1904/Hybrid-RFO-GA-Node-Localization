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
import numpy as np
import time
import random

# Step 1: Define Parameters
sensor_nodes = 600  # Increased from 300 to 600
anchor_nodes_list = [i * 10 for i in range(1, 11)]  # 10 to 100 anchors
area_size = (300, 300)  # Deployment area 300x300 m²
transmission_range = [i * 5 for i in range(2, 9)]  # 10 to 40 meters
iterations = 100
noise_variance = [2, 6]  # Two sets of experiments
mutation_probability = 0.05  # For GA & Hybrid RFO-GA
population_size = 20  # Population for metaheuristic algorithms

# Initialize sensor and anchor nodes randomly
def initialize_nodes(num_nodes, area):
    return np.random.uniform(0, area[0], (num_nodes, 2))

# Compute Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

# Fitness function (Minimize MLE)
def fitness_function(estimated_positions, true_positions):
    return np.mean([np.linalg.norm(est - true) for est, true in zip(estimated_positions, true_positions)])

# Genetic Algorithm (GA) for node localization
def genetic_algorithm(nodes, anchors, noise):
    best_solution = np.mean(anchors, axis=0)  # Placeholder: Mean position as initial guess
    best_mle = fitness_function([best_solution] * len(nodes), nodes)
    return [best_solution] * len(nodes), best_mle

# Red Fox Optimization (RFO) for node localization
def red_fox_optimization(nodes, anchors, noise):
    best_solution = np.median(anchors, axis=0)  # Placeholder: Median-based estimation
    best_mle = fitness_function([best_solution] * len(nodes), nodes)
    return [best_solution] * len(nodes), best_mle

# Hybrid RFO-GA for node localization
def hybrid_rfo_ga(nodes, anchors, noise):
    ga_positions, ga_mle = genetic_algorithm(nodes, anchors, noise)
    rfo_positions, rfo_mle = red_fox_optimization(nodes, anchors, noise)
    best_positions = [(ga + rfo) / 2 for ga, rfo in zip(ga_positions, rfo_positions)]
    best_mle = fitness_function(best_positions, nodes)
    return best_positions, best_mle

# Step 2: Run simulations for GA, RFO, Hybrid RFO-GA
def run_simulation():
    results = {'GA': [], 'RFO': [], 'Hybrid RFO-GA': []}

    for algo in results.keys():
        for _ in range(iterations):
            start_time = time.time()
            nodes = initialize_nodes(sensor_nodes, area_size)
            anchors = initialize_nodes(random.choice(anchor_nodes_list), area_size)
            noise = random.choice(noise_variance)

            if algo == 'GA':
                localized, mle = genetic_algorithm(nodes, anchors, noise)
            elif algo == 'RFO':
                localized, mle = red_fox_optimization(nodes, anchors, noise)
            else:
                localized, mle = hybrid_rfo_ga(nodes, anchors, noise)

            execution_time = time.time() - start_time
            results[algo].append({'MLE': mle, 'Computation Time': execution_time, 'Localized Nodes': len(localized)})

    return results

# Run the simulation
simulation_results = run_simulation()

# Display results
for algo, runs in simulation_results.items():
    avg_mle = np.mean([r['MLE'] for r in runs])
    avg_time = np.mean([r['Computation Time'] for r in runs])
    avg_localized_nodes = np.mean([r['Localized Nodes'] for r in runs])

    print(f"\nAlgorithm: {algo}")
    print(f"Avg Mean Localization Error (MLE): {avg_mle:.4f}")
    print(f"Avg Computation Time: {avg_time:.4f} sec")
    print(f"Avg Number of Localized Nodes: {avg_localized_nodes:.2f}")
