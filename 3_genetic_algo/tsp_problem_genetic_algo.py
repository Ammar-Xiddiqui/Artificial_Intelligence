# Genetic Algorithm for the Traveling Salesman Problem (TSP)

import random
import numpy as np

# Step 1: Create a Distance Matrix
def create_distance_matrix(num_cities):
    matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(matrix, 0)  # Distance from a city to itself is 0
    return matrix

# Step 2: Initialize population with random tours
def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        tour = list(range(num_cities))
        random.shuffle(tour)
        population.append(tour)
    return population

# Step 3: Calculate total distance of a tour
def calculate_distance(tour, distance_matrix):
    distance = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]  # Return to the starting city
        distance += distance_matrix[from_city][to_city]
    return distance

# Step 4: Evaluate fitness for each tour
def evaluate_fitness(population, distance_matrix):
    fitness_scores = []
    for tour in population:
        distance = calculate_distance(tour, distance_matrix)
        fitness = 1 / distance  # Shorter tours have higher fitness
        fitness_scores.append(fitness)
    return fitness_scores

# Step 5: Select parents using tournament selection
def select_parents(population, fitness_scores):
    selected_parents = []
    for _ in range(2):
        participants = random.sample(list(zip(population, fitness_scores)), 3)
        participants.sort(key=lambda x: x[1], reverse=True)  # Higher fitness wins
        selected_parents.append(participants[0][0])
    return selected_parents

# Step 6: Crossover to create offspring (Ordered Crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    fill_values = [city for city in parent2 if city not in child]
    pointer = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[pointer]
            pointer += 1
    return child

# Step 7: Mutation (Swap Mutation)
def mutate(tour, mutation_rate=0.01):
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

# Step 8: Genetic Algorithm for TSP
def genetic_algorithm_tsp(distance_matrix, pop_size, num_generations):
    num_cities = len(distance_matrix)
    population = initialize_population(pop_size, num_cities)
    best_tour = None
    best_distance = float('inf')
    
    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(population, distance_matrix)
        new_population = []
        
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population
        
        # Track the best tour
        for tour in population:
            distance = calculate_distance(tour, distance_matrix)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
        
        print(f"Generation {generation + 1}: Best Distance = {best_distance}")
    
    print("\nOptimal Tour:", best_tour)
    print("Shortest Distance:", best_distance)

# Example usage
if __name__ == "__main__":
    num_cities = 10
    pop_size = 100
    num_generations = 500
    
    distance_matrix = create_distance_matrix(num_cities)
    genetic_algorithm_tsp(distance_matrix, pop_size, num_generations)
