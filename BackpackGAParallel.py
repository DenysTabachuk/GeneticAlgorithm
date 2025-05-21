from BackpackGA import BackpackGA 
import random
import multiprocessing
import time
from threading import Thread

class BackpackGAParralel(BackpackGA):
    def island_worker(self, id, items, max_weight, population, generations, migration_queue, return_queue, migration_size):
        self.log(f"[Острів {id}] Стартує. Кількість поколінь: {generations}")
        for gen in range(generations):
            population = self.evolve_population(population, items, max_weight, 1)
            if gen % 10 == 0:
                best = sorted(population, key=lambda ind: self.fitness(ind), reverse=True)[:migration_size]
                migration_queue.put((id, best))
        return_queue.put((id, population))
        self.log(f"[Острів {id}] Завершено. Надіслав результат.")

    def run(self, num_islands=4):
        num_items = len(self.items)
        migration_queue = multiprocessing.Queue()
        return_queue = multiprocessing.Queue()
        processes = []

        island_pop_size = self.population_size // num_islands
        migration_size = max(1, island_pop_size // 10)
        populations = [[self.create_individual(num_items) for _ in range(island_pop_size)] for _ in range(num_islands)]

        migration_active = multiprocessing.Event()
        migration_active.set()

        def migration_worker():
            self.log("[Міграційний процес] Стартував")
            while migration_active.is_set():
                while not migration_queue.empty():
                    try:
                        island_id, migrants = migration_queue.get_nowait()
                        target_island = (island_id + random.randint(1, num_islands-1)) % num_islands
                        populations[target_island].extend(migrants)
                        populations[target_island] = populations[target_island][:island_pop_size]
                        self.log(f"[Міграція] {len(migrants)} індивідів з острова {island_id} → {target_island}")
                    except:
                        continue
                time.sleep(0.2)
            self.log("[Міграційний процес] Завершився")

        for i in range(num_islands):
            p = multiprocessing.Process(
                target=self.island_worker,
                args=(i, self.items, self.max_weight, populations[i], self.generations, migration_queue, return_queue, migration_size)
            )
            processes.append(p)
            p.start()

        migration_thread = Thread(target=migration_worker)
        migration_thread.start()

        self.log("Очікування завершення обчислень...")
        final_populations = {}
        for _ in range(num_islands):
            idx, pop = return_queue.get()
            final_populations[idx] = pop

        self.log("Всі острови завершили роботу")

        migration_active.clear()
        migration_thread.join()

        for i, p in enumerate(processes):
            p.join(timeout=2)
            if p.is_alive():
                self.log(f"[⚠️] Острів {i} досі активний. Примусово завершується.")
                p.terminate()
                p.join()

        self.log("Обробка результатів...")
        best_overall = max(
            (ind for pop in final_populations.values() for ind in pop),
            key=lambda ind: self.fitness(ind)
        )
        total_weight = sum(self.items[i][0] for i in range(num_items) if best_overall[i] == 1)

        self.log("\nПаралельний алгоритм:")
        self.log("Цінність: " + str(self.fitness(best_overall)))
        self.log("Вага: " + str(total_weight))
        return best_overall
