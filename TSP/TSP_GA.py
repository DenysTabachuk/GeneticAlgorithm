import random
from typing import List, Tuple
import math


class TSP_GA:
    def __init__(self, cities: List[Tuple[float, float]], population_size=50, generations=100, mutation_rate=0.1, verbose=False):
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.verbose = verbose
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
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def fitness(self, individual: List[int]) -> float:
        total_distance = 0
        for i in range(len(individual)):
            city_a = individual[i]
            city_b = individual[(i + 1) % self.num_cities]  # повернення в початкове місто
            total_distance += self.distance(city_a, city_b)
        return total_distance

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        # Order Crossover (OX)
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [None] * self.num_cities

        # Копіюємо сегмент з першого батька
        child[start:end+1] = parent1[start:end+1]

        # Заповнюємо решту з другого батька, зберігаючи порядок і уникаючи повторів
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

    def evolve_population(self, population: List[List[int]]) -> List[List[int]]:
        new_population = []
        population.sort(key=lambda ind: self.fitness(ind))  # сортуємо за зростанням відстані (кращий - менший)
        elite = population[:2]

        while len(new_population) < len(population):
            p1, p2 = random.choices(elite + population, k=2)
            child = self.mutate(self.crossover(p1, p2))
            new_population.append(child)

        return new_population

    def run(self):
        population = self.population
        for gen in range(self.generations):
            population = self.evolve_population(population)
            if self.verbose and gen % 10 == 0:
                best = min(population, key=lambda ind: self.fitness(ind))
                self.log(f"Покоління {gen}: найкраща відстань = {self.fitness(best):.2f}")

        best_individual = min(population, key=lambda ind: self.fitness(ind))
        best_distance = self.fitness(best_individual)

        self.log("\nРезультат TSP GA:")
        self.log(f"Найкоротший шлях: {best_individual}")
        self.log(f"Довжина шляху: {best_distance:.2f}")

        return best_individual, best_distance


