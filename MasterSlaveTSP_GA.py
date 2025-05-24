import random
from typing import List
from multiprocessing import Pool, cpu_count


class MasterSlaveTSP_GA:
    def __init__(self, cities: List[tuple], population_size=50, generations=100, mutation_rate=0.1, num_processes=None, verbose=False):
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.num_processes = num_processes or cpu_count()
        self.population = [self.create_individual() for _ in range(population_size)]

    def log(self, msg):
        if self.verbose:
            print(msg)

    def create_individual(self) -> List[int]:
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def distance(self, city1: int, city2: int) -> float:
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    def fitness(self, individual: List[int]) -> float:
        total_distance = 0
        for i in range(len(individual)):
            city_a = individual[i]
            city_b = individual[(i + 1) % self.num_cities]
            total_distance += self.distance(city_a, city_b)
        return total_distance

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        # Order Crossover (OX)
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [None] * self.num_cities
        child[start:end+1] = parent1[start:end+1]

        current_pos = (end + 1) % self.num_cities
        for city in parent2:
            if city not in child:
                child[current_pos] = city
                current_pos = (current_pos + 1) % self.num_cities

        return child

    def mutate(self, individual: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def _fitness_wrapper(self, individual):
        return self.fitness(individual)

    def _child_generator(self, parents):
        p1, p2 = parents
        return self.mutate(self.crossover(p1, p2))

    def evolve_population(self, population: List[List[int]], generations: int) -> List[List[int]]:
        with Pool(processes=self.num_processes) as pool:
            for _ in range(generations):
                # Паралельне обчислення fitness
                fitness_scores = pool.map(self._fitness_wrapper, population)

                # Сортуємо за зростанням (кращі мають менший fitness)
                sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]

                elite = sorted_population[:2]

                mating_pool = elite + sorted_population
                num_children = len(population)

                # Вибираємо пари батьків
                parent_pairs = [random.choices(mating_pool, k=2) for _ in range(num_children)]

                # Паралельне створення потомків
                population = pool.map(self._child_generator, parent_pairs)

        return population

    def run(self, num_processes=None):
        if num_processes:
            self.num_processes = num_processes

        population = self.population

        population = self.evolve_population(population, self.generations)

        best_individual = min(population, key=lambda ind: self.fitness(ind))
        best_distance = self.fitness(best_individual)

        self.log("\nMaster-Slave алгоритм для TSP:")
        self.log(f"Найкоротший шлях: {best_individual}")
        self.log(f"Довжина шляху: {best_distance:.2f}")

        return best_individual, best_distance
