from BackpackGA import BackpackGA
import multiprocessing
import time
from threading import Thread
from typing import List, Tuple, Dict

class BackpackGAIslandModel(BackpackGA):
    def _island_worker(
        self,
        id: int,
        population: List[List[int]],
        in_queue: multiprocessing.Queue,
        migration_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        migration_size: int,
        migration_interval: int,
        accepted_migrations_counter: multiprocessing.Value = None
    ) -> None:
        self._log(f"[Острів {id}] Стартує.")
        pop_size = len(population)

        for gen in range(self.generations):
            # Отримуємо мігрантів, якщо є
            try:
                migrants: List[List[int]] = in_queue.get_nowait()
                population.extend(migrants)
                population = sorted(population, key=self._fitness, reverse=True)[:pop_size]
                self._log(f"[Острів {id}] Прийняв {len(migrants)} мігрантів")

                if accepted_migrations_counter is not None:
                    with accepted_migrations_counter.get_lock():
                        accepted_migrations_counter.value += 1
            except:
                pass

            # Еволюція на одне покоління
            population = list(self._evolve_population(population, 1))

            # Кожні migration_interval поколінь надсилаємо мігрантів
            if gen % migration_interval == 0:
                best = sorted(population, key=self._fitness, reverse=True)[:migration_size]
                migration_queue.put((id, best))

        result_queue.put((id, population))
        self._log(f"[Острів {id}] Завершено. Надіслав результат.")
        return

    def _migration_worker(
        self,
        in_queues: List[multiprocessing.Queue],
        num_islands: int,
        migration_queue: multiprocessing.Queue,
        migration_active: multiprocessing.Event,
        migration_count: multiprocessing.Value
    ) -> None:
        self._log("[Міграційний процес] Стартував")
        while migration_active.is_set():
            try:
                island_id, migrants = migration_queue.get_nowait()
                target_island = (island_id + 1) % num_islands
                in_queues[target_island].put(migrants)

                with migration_count.get_lock():
                    migration_count.value += 1

                self._log(f"[Міграція] {len(migrants)} індивідів з острова {island_id} → {target_island}")
            except:
                time.sleep(0.2)
        self._log("[Міграційний процес] Завершився")

    def run(
        self,
        num_islands: int ,
        migration_interval: int = 10
    ) -> Tuple[List[int], int, int]:
        min_island_pop = 4
        while self.population_size // num_islands < min_island_pop:
            num_islands -= 1

        num_items: int = len(self.items)
        island_pop_size: int = self.population_size // num_islands
        migration_size: int = max(1, island_pop_size // 10)

        populations: List[List[List[int]]] = [
            [self._create_individual(num_items) for _ in range(island_pop_size)]
            for _ in range(num_islands)
        ]

        in_queues: List[multiprocessing.Queue] = [multiprocessing.Queue() for _ in range(num_islands)]
        migration_queue: multiprocessing.Queue = multiprocessing.Queue()
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        migration_active: multiprocessing.Event = multiprocessing.Event()
        migration_active.set()
        migration_count: multiprocessing.Value = multiprocessing.Value('i', 0)
        accepted_migrations_count: multiprocessing.Value = multiprocessing.Value('i', 0)

        processes: List[multiprocessing.Process] = []

        for i in range(num_islands):
            p = multiprocessing.Process(
                target=self._island_worker,
                args=(
                    i,
                    populations[i],
                    in_queues[i],
                    migration_queue,
                    result_queue,
                    migration_size,
                    migration_interval,
                    accepted_migrations_count
                )
            )
            processes.append(p)
            p.start()

        migration_thread = Thread(
            target=self._migration_worker,
            args=(in_queues, num_islands, migration_queue, migration_active, migration_count)
        )
        migration_thread.start()

        final_populations: Dict[int, List[List[int]]] = {}
        for _ in range(num_islands):
            idx, pop = result_queue.get()
            final_populations[idx] = pop

        self._log("Всі острови завершили роботу")

        migration_active.clear()
        migration_thread.join()

        self._log("Обробка результатів...")

        best_individual: List[int] = max(
            (ind for pop in final_populations.values() for ind in pop),
            key=self._fitness
        )

        best_value: int = sum(self.items[i][1] for i in range(num_items) if best_individual[i] == 1)
        best_weight: int = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        self._log(f"Загальна кількість міграцій: {migration_count.value}")
        self._log(f"Загальна кількість прийнятих міграцій: {accepted_migrations_count.value}")

        return best_individual, best_value, best_weight
