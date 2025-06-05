from BackpackGA import BackpackGA
import multiprocessing
import time
from threading import Thread
from typing import List, Tuple, Dict


class BackpackGAIslandModel(BackpackGA):
    # Окремий процес, що виконує еволюцію на одному острові
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
            # Перевірка черги на наявність мігрантів
            try:
                migrants: List[List[int]] = in_queue.get_nowait()
                population.extend(migrants)
                population = sorted(population, key=self._fitness, reverse=True)[:pop_size]
                self._log(f"[Острів {id}] Прийняв {len(migrants)} мігрантів")

                # Підрахунок прийнятих міграцій
                if accepted_migrations_counter is not None:
                    with accepted_migrations_counter.get_lock():
                        accepted_migrations_counter.value += 1
            except:
                pass

            # Еволюція на одне покоління
            population = list(self._evolve_population(population, 1))

            # Періодична міграція найкращих особин
            if gen % migration_interval == 0:
                best = sorted(population, key=self._fitness, reverse=True)[:migration_size]
                migration_queue.put((id, best))

        # Надсилання фінальної популяції
        result_queue.put((id, population))
        self._log(f"[Острів {id}] Завершено. Надіслав результат.")
        return

    # Окремий потік, що керує міжострівною міграцією
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
                # Міграція до наступного острова по колу
                target_island = (island_id + 1) % num_islands
                in_queues[target_island].put(migrants)

                # Підрахунок загальної кількості міграцій
                with migration_count.get_lock():
                    migration_count.value += 1

                self._log(f"[Міграція] {len(migrants)} індивідів з острова {island_id} → {target_island}")
            except:
                time.sleep(0.2)
        self._log("[Міграційний процес] Завершився")

    # Основний метод запуску Island Model
    def run(
        self,
        num_threads: int,
        migration_interval: int = 10
    ) -> Tuple[List[int], int, int]:

        if num_threads < 2:
            raise ValueError("Необхідно принаймні 2 потоки: 1 для міграцій, решта для островів.")

        num_islands = num_threads - 1

        # Перевірка, щоб популяція була достатньою для розподілу
        min_island_pop = 4
        while self.population_size // num_islands < min_island_pop:
            num_islands -= 1

        if num_islands < 1:
            raise ValueError("Неможливо розподілити популяцію по островах. Збільште розмір популяції або зменшіть кількість потоків.")

        num_items: int = len(self.items)
        island_pop_size: int = self.population_size // num_islands
        migration_size: int = max(1, island_pop_size // 10)

        # Створення початкових популяцій для кожного острова
        populations: List[List[List[int]]] = [
            [self._create_individual(num_items) for _ in range(island_pop_size)]
            for _ in range(num_islands)
        ]

        # Черги для прийому мігрантів на кожному острові (кожен острів має окрему чергу)
        in_queues: List[multiprocessing.Queue] = [multiprocessing.Queue() for _ in range(num_islands)]
        # Загальна черга, куди острови надсилають своїх найкращих особин для міграції
        migration_queue: multiprocessing.Queue = multiprocessing.Queue()
        # Черга для збору фінальних популяцій від усіх островів після завершення еволюції
        result_queue: multiprocessing.Queue = multiprocessing.Queue()


        # Синхронізація і підрахунки
        migration_active: multiprocessing.Event = multiprocessing.Event() # сигнальний об’єкт, спеціально створений для безпечної та ефективної синхронізації між процесами.


        migration_active.set() # set() migration_active = True


        migration_count: multiprocessing.Value = multiprocessing.Value('i', 0) # аналог атомарної змінної з блокуванням, 'i' — ціле число (int) 
        accepted_migrations_count: multiprocessing.Value = multiprocessing.Value('i', 0)

        # Запуск островів у окремих процесах
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

        # Запуск окремого потоку для координації міграцій
        migration_thread = Thread(
            target=self._migration_worker,
            args=(in_queues, num_islands, migration_queue, migration_active, migration_count)
        )
        migration_thread.start()

        # Збір результатів з кожного острова
        final_populations: Dict[int, List[List[int]]] = {}
        for _ in range(num_islands):
            idx, pop = result_queue.get()
            final_populations[idx] = pop

        self._log("Всі острови завершили роботу")

        # Завершення міграцій
        migration_active.clear()
        migration_thread.join()

        self._log("Обробка результатів...")

        # Пошук найкращого індивіда серед усіх островів
        best_individual: List[int] = max(
            (ind for pop in final_populations.values() for ind in pop),
            key=self._fitness
        )

        # Обрахунок ваги та цінності найкращого рішення
        best_value: int = sum(self.items[i][1] for i in range(num_items) if best_individual[i] == 1)
        best_weight: int = sum(self.items[i][0] for i in range(num_items) if best_individual[i] == 1)

        # Вивід статистики
        self._log(f"Загальна кількість міграцій: {migration_count.value}")
        self._log(f"Загальна кількість прийнятих міграцій: {accepted_migrations_count.value}")

        return best_individual, best_value, best_weight
