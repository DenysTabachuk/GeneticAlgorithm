import random
import time
import pandas as pd
import BackpackGA as sequntialGA
import BackpackGAIslandModel as parallelGA
import BackpackGAMasterSlave as masterSlaveGA

import os

def get_logical_cores() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1

def generate_items(num_items=100):
    items = []
    total_weight = 0
    for _ in range(num_items):
        weight = random.randint(1, 20)
        value = random.randint(5, 100)
        items.append((weight, value))
        total_weight += weight
    return items, total_weight


def run_comparison_tests(
    population_sizes=[100, 200, 300, 400, 500],
    generations_list=[100, 200, 300],
    num_threads_list=[4, 8, 12, 16],
    num_items=100,
    mutation_rate=0.1,
    num_cores=12,
    num_runs=4  
):
    random.seed(42)
    items, total_weight = generate_items(num_items)
    max_weight = int(total_weight * 0.4)

    results = []

    for population_size in population_sizes:
        for generations in generations_list:
            for num_threads in num_threads_list:

                seq_times, par_times, ms_times = [], [], []
                val_seq_list, val_par_list, val_ms_list = [], [], []

                for _ in range(num_runs):
                    # === Послідовний алгоритм ===
                    gaSeq = sequntialGA.BackpackGA(
                        items,
                        max_weight,
                        population_size=population_size,
                        generations=generations,
                        mutation_rate=mutation_rate,
                        verbose=False,
                    )
                    start_time = time.time()
                    _, val_seq, _ = gaSeq.run()
                    seq_times.append(time.time() - start_time)
                    val_seq_list.append(val_seq)

                    # === Island Model ===
                    gaPar = parallelGA.BackpackGAIslandModel(
                        items,
                        max_weight,
                        population_size=population_size,
                        generations=generations,
                        mutation_rate=mutation_rate,
                        verbose=False,
                    )
                    start_time = time.time()
                    _, val_par, _ = gaPar.run(num_threads)
                    par_times.append(time.time() - start_time)
                    val_par_list.append(val_par)

                    # === Master-Slave ===
                    gaMaster = masterSlaveGA.BackpackGAMasterSlave(
                        items,
                        max_weight,
                        population_size=population_size,
                        generations=generations,
                        mutation_rate=mutation_rate,
                        verbose=False,
                    )
                    start_time = time.time()
                    _, val_master, _ = gaMaster.run(num_cores)
                    ms_times.append(time.time() - start_time)
                    val_ms_list.append(val_master)

                # === Усереднені метрики ===
                avg_seq_time = sum(seq_times) / num_runs
                avg_par_time = sum(par_times) / num_runs
                avg_ms_time = sum(ms_times) / num_runs

                avg_val_seq = sum(val_seq_list) / num_runs
                avg_val_par = sum(val_par_list) / num_runs
                avg_val_ms = sum(val_ms_list) / num_runs

                island_speedup = avg_seq_time / avg_par_time if avg_par_time > 0 else float('inf')
                ms_speedup = avg_seq_time / avg_ms_time if avg_ms_time > 0 else float('inf')

                island_eff = island_speedup / num_cores
                ms_eff = ms_speedup / num_cores

                results.append({
                    "Pop.Size": population_size,
                    "Generations": generations,
                    "Threads": num_threads,
                    "Max Weight": max_weight,
                    "Seq.Time (s)": round(avg_seq_time, 2),
                    "Island.Time (s)": round(avg_par_time, 2),
                    "MS.Time (s)": round(avg_ms_time, 2),
                    "Seq Value": round(avg_val_seq, 1),
                    "Island Value": round(avg_val_par, 1),
                    "MS Value": round(avg_val_ms, 1),
                    "Island Speedup": round(island_speedup, 2),
                    "MS Speedup": round(ms_speedup, 2),
                    "Island Eff.": round(island_eff, 2),
                    "MS Eff.": round(ms_eff, 2),
                })

                print(
                    f"Done: Pop={population_size}, Gen={generations}, Threads={num_threads} "
                    f"(Avg over {num_runs} runs) → Seq={avg_seq_time:.2f}s, Island={avg_par_time:.2f}s, MS={avg_ms_time:.2f}s"
                )

    df = pd.DataFrame(results)
    df.to_csv("comparison_results_avg.csv", index=False)
    print("\n=== Середні результати після кількох прогонів ===\n")
    print(df.to_string(index=False))




if __name__ == "__main__":
    print(f"Кількість логічних ядер: {get_logical_cores()}")

    run_comparison_tests()


# if __name__ == "__main__":
#     print(f"Кількість логічних ядер: {get_logical_cores()}")

#     # === Параметри ===
#     population_size = 500
#     generations = 100
#     num_threads = 12 
#     mutation_rate = 0.1
#     num_items = 100

#     random.seed(42)
#     items, total_weight = generate_items(num_items)
#     max_weight = int(total_weight * 0.4)

#     # === Послідовний алгоритм ===
#     ga_seq = sequntialGA.BackpackGA(
#         items,
#         max_weight,
#         population_size=population_size,
#         generations=generations,
#         mutation_rate=mutation_rate,
#         verbose=False,
#     )
#     start = time.time()
#     best_seq, val_seq, wt_seq = ga_seq.run()
#     time_seq = time.time() - start

#     # === Island Model ===
#     ga_island = parallelGA.BackpackGAIslandModel(
#         items,
#         max_weight,
#         population_size=population_size,
#         generations=generations,
#         mutation_rate=mutation_rate,
#         verbose=False,
#     )
#     start = time.time()
#     best_island, val_island, wt_island = ga_island.run(num_threads)
#     time_island = time.time() - start

#     # === Master-Slave ===
#     ga_ms = masterSlaveGA.BackpackGAMasterSlave(
#         items,
#         max_weight,
#         population_size=population_size,
#         generations=generations,
#         mutation_rate=mutation_rate,
#         verbose=False,
#     )
#     start = time.time()
#     best_ms, val_ms, wt_ms = ga_ms.run(num_threads)
#     time_ms = time.time() - start

#     # === Вивід результатів ===
#     print("\n=== Порівняння алгоритмів ===")
#     print(f"Популяція: {population_size}, Покоління: {generations}")
#     print(f"Макс. вага: {max_weight}")

#     print(f"\n[Послідовний]")
#     print(f"Час: {time_seq:.2f} с, Вага: {wt_seq}, Цінність: {val_seq}")

#     print(f"\n[Island Model] (Потоки: {num_threads})")
#     print(f"Час: {time_island:.2f} с, Вага: {wt_island}, Цінність: {val_island}")

#     print(f"\n[Master-Slave] (Ядер: {num_threads})")
#     print(f"Час: {time_ms:.2f} с, Вага: {wt_ms}, Цінність: {val_ms}")

#     # === Прискорення та ефективність ===
#     print("\n=== Прискорення та ефективність ===")
#     speedup_island = time_seq / time_island if time_island > 0 else float('inf')
#     speedup_ms = time_seq / time_ms if time_ms > 0 else float('inf')
#     eff_island = speedup_island / get_logical_cores()
#     eff_ms = speedup_ms / get_logical_cores()

#     print(f"Island Speedup: {speedup_island:.2f}, Efficiency: {eff_island:.2f}")
#     print(f"Master-Slave Speedup: {speedup_ms:.2f}, Efficiency: {eff_ms:.2f}")
