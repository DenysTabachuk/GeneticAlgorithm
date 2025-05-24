import random
from typing import List
import time
import BackpackGA as sequntialGA
import BackpackGAIslandModel as parallelGA
import BackpackGAMasterSlave as masterSlaveGA


def generate_items(num_items=100):  
    items = []
    max_weight = 0
    for _ in range(num_items):
        weight = random.randint(1, 20)  
        value = random.randint(5, 100)  
        items.append((weight, value))
        max_weight += weight
    return items


def test_sequential(runs=5, **kwargs):
    _test_against_known_optimum(sequntialGA.BackpackGA, runs=runs, label="Послідовний", **kwargs)

def test_parallel(runs=5, **kwargs):
    _test_against_known_optimum(parallelGA.BackpackGAIslandModel, runs=runs, label="Паралельний", **kwargs)

def _test_against_known_optimum(algorithm_class, runs=5, label="", **kwargs):
    known_items = [(2, 3), (1, 2), (3, 4), (2, 2)]
    max_weight = 6
    expected_value = 9

    success_count = 0
    for i in range(runs):
        ga = algorithm_class(known_items, max_weight, **kwargs)
        best = ga.run()
        fitness_val = ga.fitness(best)
        if fitness_val == expected_value:
            success_count += 1

    print(f"\n{label} — збіг з оптимумом: {success_count} з {runs} ({(success_count/runs)*100:.1f}%)")


if __name__ == "__main__":
    # test_sequential(runs=10, verbose=False)
    # test_parallel(runs=10, verbose=False)
    random.seed(42)

    items = generate_items(num_items=1000)
    max_weight = 5000

    # Запуск послідовного алгоритму
    start_time = time.time()
    gaSeq = sequntialGA.BackpackGA(items, max_weight, population_size=20, generations=100, mutation_rate=0.1)
    best_solution, value, weight = gaSeq.run()
    seq_time = time.time() - start_time

    # Запуск паралельного алгоритму Island Model
    start_time = time.time()
    gaPar = parallelGA.BackpackGAIslandModel(items, max_weight, population_size=20, generations=100, mutation_rate=0.1, verbose=False)
    best_solution_par, value_par, weight_par = gaPar.run(12)
    par_time = time.time() - start_time

    # Запуск паралельного алгоритму Master-Slave
    start_time = time.time()
    gaMaster = masterSlaveGA.BackpackGAMasterSlave(items, max_weight, population_size=20, generations=100, mutation_rate=0.1)
    best_sol_master, val_master, wt_master = gaMaster.run(12)
    master_time = time.time() - start_time


    # Вивід результатів
    print("\n=== Результати виконання алгоритмів ===\n")

    print("Послідовний алгоритм:")
    print(f"  Цінність: {value}")
    print(f"  Вага:     {weight}")
    print(f"  Час:     {seq_time:.2f} сек\n")

    print("Паралельний алгоритм Island Model:")
    print(f"  Цінність: {value_par}")
    print(f"  Вага:     {weight_par}")
    print(f"  Час:     {par_time:.2f} сек")
    print(f"  Прискорення: {seq_time/par_time:.2f}x\n")

    print("Паралельний алгоритм Master-Slave:")
    print(f"  Цінність: {val_master}")
    print(f"  Вага:     {wt_master}")
    print(f"  Час:     {master_time:.2f} сек")
    print(f"  Прискорення: {seq_time/master_time:.2f}x\n")



