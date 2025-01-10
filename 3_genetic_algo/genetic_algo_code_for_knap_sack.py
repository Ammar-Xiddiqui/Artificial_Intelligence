import random

# Item details (weight, value)
items = [
    (10, 60),  # Item 1
    (20, 100), # Item 2
    (30, 120), # Item 3
    (15, 75),  # Item 4
    (25, 90)   # Item 5
]

knapsack_weight_limit = 50

class Chromosome:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = [random.randint(0, 1) for _ in range(len(items))]
        else:
            self.genes = genes
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        total_weight = total_value = 0
        for gene, (weight, value) in zip(self.genes, items):
            if gene == 1:
                total_weight += weight
                total_value += value
        return total_value if total_weight <= knapsack_weight_limit else 0

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = self.initialize_population()

    def initialize_population(self):
        return [Chromosome() for _ in range(self.population_size)]

    def selection(self):
        max_fitness = sum(chromo.fitness for chromo in self.population)
        pick = random.uniform(0, max_fitness)
        current = 0
        for chromo in self.population:
            current += chromo.fitness
            if current > pick:
                return chromo

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1.genes) - 1)
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(child1_genes), Chromosome(child2_genes)
        return parent1, parent2

    def mutate(self, chromosome):
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_rate:
                chromosome.genes[i] = 1 - chromosome.genes[i]
        chromosome.fitness = chromosome.calculate_fitness()

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = sorted(new_population, key=lambda c: c.fitness, reverse=True)[:self.population_size]

    def get_best_solution(self):
        return max(self.population, key=lambda c: c.fitness)

# Parameters: population size, mutation rate, crossover rate, generations
ga = GeneticAlgorithm(population_size=10, mutation_rate=0.01, crossover_rate=0.7, generations=20)

ga.evolve()
best = ga.get_best_solution()

print("Best solution:")
print(f"Genes: {best.genes}")
print(f"Fitness (Total Value): {best.fitness}")
