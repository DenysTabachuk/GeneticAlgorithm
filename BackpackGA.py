import random
from typing import List
import time

class BackpackGA:
    def __init__(self, items: List[tuple], max_weight: int, population_size=20, generations=50, mutation_rate=0.1, verbose=False):
        self.items = items
        self.max_weight = max_weight
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.population = [self.create_individual(len(items)) for _ in range(population_size)]

    def log(self, msg):
        if self.verbose:
            print(msg)

    def create_individual(self, num_items: int) -> List[int]:
        return [random.randint(0, 1) for _ in range(num_items)]

    def crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        return [gene1 if random.random() < 0.7 else gene2 for gene1, gene2 in zip(p1, p2)]

    def mutate(self, individual: List[int], mutation_rate=None) -> List[int]:
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]

    def fitness(self, individual: List[int]) -> int:
        # time.sleep(0.005)  # Simulate a time-consuming fitness calculation
        total_weight = sum(self.items[i][0] for i in range(len(self.items)) if individual[i] == 1)
        total_value = sum(self.items[i][1] for i in range(len(self.items)) if individual[i] == 1)
        return total_value if total_weight <= self.max_weight else 0

    def evolve_population(self, population: List[List[int]], items: List[tuple], max_weight: int, generations: int) -> List[List[int]]:
        def tournament_selection(population: List[List[int]], k=3) -> List[int]:
            selected = random.sample(population, k)
            return max(selected, key=self.fitness)

        for gen in range(generations):
            population.sort(key=lambda ind: self.fitness(ind), reverse=True)
            elite = population[:2]  # топ-2 особини

            new_population = elite.copy()  # Явне збереження еліти

            while len(new_population) < len(population):
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)
                child = self.mutate(self.crossover(p1, p2))
                new_population.append(child)

            population = new_population

            if self.verbose:
                best_fit = self.fitness(population[0])
                self.log(f"Покоління {gen+1}: найкраща цінність = {best_fit}")

        return population


    def run(self):
        num_items = len(self.items)
        population = [self.create_individual(num_items) for _ in range(self.population_size)]

        for _ in range(self.generations):
            population = self.evolve_population(population, self.items, self.max_weight, 1)

        best_individual = max(population, key=lambda ind: self.fitness(ind))
        total_weight = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        self.log("\nПослідовний алгоритм:")
        self.log("Цінність: " + str(self.fitness(best_individual)))
        self.log("Вага: " + str(total_weight))

        return best_individual