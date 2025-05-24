import random
from typing import List
from multiprocessing import Pool
from BackpackGA import BackpackGA

class BackpackGAMasterSlave(BackpackGA):
    def _mutate_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        child = self._crossover(p1, p2)
        return self._mutate(child)

    def _evolve_population(self, population: List[List[int]],  num_threads: int) -> List[List[int]]:
        with Pool(processes=num_threads) as pool:
            for gen in range(self.generations):
                fitness_scores = pool.map(self._fitness, population)
                sorted_population = [ind for _, ind in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
                elite = sorted_population[:2]
                mating_pool = elite + sorted_population
                num_children = len(population)
                parent_pairs = [random.choices(mating_pool, k=2) for _ in range(num_children)]
                population = pool.starmap(self._mutate_crossover, parent_pairs)

                best_fit = self._fitness(population[0])
                self._log(f"Покоління {gen+1}/{self.generations}: найкращий fitness = {best_fit:.4f}")

        return population

    def run(self, num_threads: int, verbose: bool = False):
        self.verbose = verbose
        num_items = len(self.items)
        population = [self._create_individual(num_items) for _ in range(self.population_size)]
        final_population = self._evolve_population(population, num_threads)
        best_individual = max(final_population, key=lambda ind: self._fitness(ind))
        best_value = sum(self.items[i][1] for i in range(num_items) if best_individual[i] == 1)
        best_weight = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)
        return best_individual, best_value, best_weight
