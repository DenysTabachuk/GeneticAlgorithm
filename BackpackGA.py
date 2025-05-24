import random
from typing import List

class BackpackGA:
    def __init__(self, items: List[tuple], max_weight: int, population_size=20, generations=50, mutation_rate=0.1, verbose=False):
        self.items = items
        self.max_weight = max_weight
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.population = [self._create_individual(len(items)) for _ in range(population_size)]

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _create_individual(self, num_items: int) -> List[int]:
        return [random.randint(0, 1) for _ in range(num_items)]

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        return [gene1 if random.random() < 0.7 else gene2 for gene1, gene2 in zip(p1, p2)]

    def _mutate(self, individual: List[int], mutation_rate=None) -> List[int]:
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]

    def _fitness(self, individual: List[int]) -> int:
        total_weight = sum(self.items[i][0] for i in range(len(self.items)) if individual[i] == 1)
        total_value = sum(self.items[i][1] for i in range(len(self.items)) if individual[i] == 1)
        return total_value if total_weight <= self.max_weight else 0

    def _evolve_population(self, population: List[List[int]], generations: int) -> List[List[int]]:
        def tournament_selection(population: List[List[int]], k=3) -> List[int]:
            selected = random.sample(population, k)
            return max(selected, key=self._fitness)

        for gen in range(generations):
            population.sort(key=self._fitness, reverse=True)
            elite = population[:2]

            new_population = elite.copy()

            while len(new_population) < len(population):
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)
                child = self._mutate(self._crossover(p1, p2))
                new_population.append(child)

            population = new_population

            if self.verbose:
                best_fit = self._fitness(population[0])
                self._log(f"Покоління {gen+1}: найкраща цінність = {best_fit}")

        return population

    def run(self):
        num_items = len(self.items)
        population = [self._create_individual(num_items) for _ in range(self.population_size)]

        for _ in range(self.generations):
            population = self._evolve_population(population, 1)

        best_individual = max(population, key=lambda ind: self._fitness(ind))
        total_weight = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        self._log("\nПослідовний алгоритм:")
        self._log("Цінність: " + str(self._fitness(best_individual)))
        self._log("Вага: " + str(total_weight))

        return best_individual
