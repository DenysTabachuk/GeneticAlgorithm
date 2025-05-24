import random
from typing import List, Tuple

class BackpackGA:
    def __init__(
        self,
        items: List[Tuple[int, int]],
        max_weight: int,
        population_size: int,
        generations: int,
        mutation_rate: float,
        verbose: bool = False,
    ):
        self.items: List[Tuple[int, int]] = items
        self.max_weight: int = max_weight
        self.population_size: int = population_size
        self.generations: int = generations
        self.mutation_rate: float = mutation_rate
        self.verbose: bool = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _create_individual(self, num_items: int) -> List[int]:
        return [random.randint(0, 1) for _ in range(num_items)]

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        fitness_p1: float = self._fitness(p1)
        fitness_p2: float = self._fitness(p2)

        if fitness_p1 > fitness_p2:
            better, worse = p1, p2
        else:
            better, worse = p2, p1

        prob_better: float = 0.75
        child: List[int] = []
        for i in range(len(p1)):
            if random.random() < prob_better:
                child.append(better[i])
            else:
                child.append(worse[i])
        return child

    def _mutate(self, individual: List[int]) -> List[int]:
        return [bit if random.random() > self.mutation_rate else 1 - bit for bit in individual]

    def _fitness(self, individual: List[int]) -> float:
        total_weight: int = sum(
            self.items[i][0] for i in range(len(self.items)) if individual[i] == 1
        )
        total_value: int = sum(
            self.items[i][1] for i in range(len(self.items)) if individual[i] == 1
        )
        if total_weight > self.max_weight:
            return 0.0
        return total_value - 0.1 * total_weight

    def _evolve_population(self, population: List[List[int]], generations: int) -> List[List[int]]:
        def tournament_selection(pop: List[List[int]], k: int = 3) -> List[int]:
            selected: List[List[int]] = random.sample(pop, k)
            return max(selected, key=self._fitness)

        for gen in range(generations):
            population.sort(key=self._fitness, reverse=True)
            elite: List[List[int]] = population[:2]

            new_population: List[List[int]] = elite.copy()

            while len(new_population) < len(population):
                p1: List[int] = tournament_selection(population)
                p2: List[int] = tournament_selection(population)
                child: List[int] = self._mutate(self._crossover(p1, p2))
                new_population.append(child)

            population = new_population

            if self.verbose:
                best_fit: float = self._fitness(population[0])
                # self._log(f"Покоління {gen+1}: найкраща цінність = {best_fit}")

        return population

    def run(self) -> Tuple[List[int], int, int]:
        num_items: int = len(self.items)
        population: List[List[int]] = [self._create_individual(num_items) for _ in range(self.population_size)]

        final_population: List[List[int]] = self._evolve_population(population, self.generations)

        best_individual: List[int] = max(final_population, key=self._fitness)
        best_value: int = sum(self.items[i][1] for i in range(num_items) if best_individual[i] == 1)
        best_weight: int = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        return best_individual, best_value, best_weight
