import random
from typing import List
from multiprocessing import Pool, cpu_count
from BackpackGA import BackpackGA  


class MasterSlaveBackpackGA(BackpackGA):
    def __init__(self, *args, num_processes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = num_processes or cpu_count()

    def _fitness_wrapper(self, individual):
        return self.fitness(individual)

    def _child_generator(self, args):
        p1, p2 = args
        return self.mutate(self.crossover(p1, p2))

    def evolve_population(self, population: List[List[int]], generations: int) -> List[List[int]]:
        with Pool(processes=self.num_processes) as pool:
            for _ in range(generations):
                # Паралельне обчислення fitness
                fitness_scores = pool.map(self._fitness_wrapper, population)
                
                # Сортування популяції
                sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
                elite = sorted_population[:2]

                # Підготовка пар для кросоверу
                mating_pool = elite + sorted_population
                num_children = len(population)
                parent_pairs = [random.choices(mating_pool, k=2) for _ in range(num_children)]

                # Паралельне створення потомків
                population = pool.map(self._child_generator, parent_pairs)

        return population

    def run(self, num_processes=None):
        if num_processes:
            self.num_processes = num_processes

        num_items = len(self.items)
        population = [self.create_individual(num_items) for _ in range(self.population_size)]

        population = self.evolve_population(population, self.generations)

        best_individual = max(population, key=lambda ind: self.fitness(ind))
        total_weight = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        self.log("\nMaster-Slave алгоритм (з постійним пулом процесів):")
        self.log("Цінність: " + str(self.fitness(best_individual)))
        self.log("Вага: " + str(total_weight))

        return best_individual
