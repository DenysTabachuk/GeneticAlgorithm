from typing import List, Tuple
from multiprocessing import Pool
from BackpackGA import BackpackGA

class BackpackGAMasterSlave(BackpackGA):
    def _mutate_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        child = self._crossover(p1, p2)
        return self._mutate(child)

    def _evolve_population(self, population: List[List[int]], num_threads: int) -> List[List[int]]:
        with Pool(processes=num_threads) as pool:
            for gen in range(self.generations):
                self._log(f"\n--- Покоління {gen+1}/{self.generations} ---")
                self._log("Починаємо обчислення фітнесів у потоках...")

                # Розподіл: оцінка фітнесу
                fitness_scores = pool.map(self._fitness, population)
                self._log(f"Фітнеси обчислено (використано {num_threads} потоків).")

                # Сортування популяції
                sorted_pop_with_fit = sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
                sorted_population = [ind for _, ind in sorted_pop_with_fit]

                # Еліта (кращі індивіди)
                elite = sorted_population[:2]
                new_population = elite.copy()
                self._log(f"Еліта збережена: {len(elite)} індивіди.")

                # Формування батьківських пар
                parent_pairs = []
                while len(new_population) + len(parent_pairs) < self.population_size:
                    p1 = self._tournament_selection(sorted_population)
                    p2 = self._tournament_selection(sorted_population)
                    parent_pairs.append((p1, p2))
                self._log(f"Батьківських пар для кросоверу: {len(parent_pairs)}")

                # Розподіл: створення нащадків
                self._log("Створення нащадків у потоках...")
                children = pool.starmap(self._mutate_crossover, parent_pairs)
                self._log(f"Нащадків створено: {len(children)}")

                # Завершення популяції
                new_population.extend(children[:self.population_size - len(new_population)])
                self._log(f"Нова популяція сформована (розмір = {len(new_population)}).")
                population = new_population

                # Логування найкращого індивіда
                if self.verbose:
                    best_ind = max(population, key=self._fitness)
                    best_fit = self._fitness(best_ind)
                    best_weight = sum(self.items[i][0] for i in range(len(self.items)) if best_ind[i] == 1)
                    self._log(f"Покоління {gen+1}/{self.generations}: найкращий fitness = {best_fit:.4f}, вага = {best_weight}")
                    self._log(f"Найкращий індивід: {best_ind}")

        return population

    def run(self, num_threads: int, ) -> Tuple[List[int], int, int]:
        num_items = len(self.items)
        population = [self._create_individual(num_items) for _ in range(self.population_size)]
        final_population = self._evolve_population(population, num_threads)
        best_individual = max(final_population, key=self._fitness)
        best_value = sum(self.items[i][1] for i in range(num_items) if best_individual[i] == 1)
        best_weight = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)
        return best_individual, best_value, best_weight
