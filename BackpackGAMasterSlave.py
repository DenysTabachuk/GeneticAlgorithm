import random
from typing import List
from multiprocessing import Pool, cpu_count
from BackpackGA import BackpackGA

class BackpackGAMasterSlave(BackpackGA):
    def __init__(self, *args, num_processes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = num_processes or cpu_count()

    def _fitness_wrapper(self, individual):
        return self._fitness(individual)

    def _child_generator(self, args):
        p1, p2 = args
        return self._mutate(self._crossover(p1, p2))

    def _evolve_population(self, population: List[List[int]], generations: int) -> List[List[int]]:
        with Pool(processes=self.num_processes) as pool:
            for _ in range(generations):
                # Паралельне обчислення fitness
                fitness_scores = pool.map(self._fitness_wrapper, population)
                
                # Сортування популяції за спаданням fitness
                sorted_population = [ind for _, ind in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
                elite = sorted_population[:2]

                # Формування пар для кросоверу
                mating_pool = elite + sorted_population
                num_children = len(population)
                parent_pairs = [random.choices(mating_pool, k=2) for _ in range(num_children)]

                # Паралельне створення потомків
                population = pool.map(self._child_generator, parent_pairs)

        return population

    def run(self, num_processes=None):
        if num_processes is not None:
            self.num_processes = num_processes

        num_items = len(self.items)
        population = [self._create_individual(num_items) for _ in range(self.population_size)]

        population = self._evolve_population(population, self.generations)

        best_individual = max(population, key=lambda ind: self._fitness(ind))
        best_value = self._fitness(best_individual)
        best_weight = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        self._log("\nMaster-Slave алгоритм (з постійним пулом процесів):")
        self._log("Цінність: " + str(best_value))
        self._log("Вага: " + str(best_weight))

        return best_individual, best_value, best_weight
