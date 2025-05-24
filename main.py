import random
from typing import List
import time
import BackpackGA as sequntialGA
import BackpackGAParallel as parallelGA
import MasterSalveGa as masterSlaveGA


def generate_items(num_items=100):  
    items = []
    max_weight = 0
    for _ in range(num_items):
        weight = random.randint(1, 20)  
        value = random.randint(5, 20)  
        items.append((weight, value))
        max_weight += weight
    max_weight = int(max_weight * 0.8)
    return items, max_weight



def test_sequential(runs=5, **kwargs):
    _test_against_known_optimum(sequntialGA.BackpackGA, runs=runs, label="Послідовний", **kwargs)

def test_parallel(runs=5, **kwargs):
    _test_against_known_optimum(parallelGA.BackpackGAParralel, runs=runs, label="Паралельний", **kwargs)

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

    items, max_weight = generate_items(num_items=1000)
    max_weight = 5000

    start_time = time.time()
    gaSeq = sequntialGA.BackpackGA(items, max_weight, population_size=20, generations=10, mutation_rate=0.1)
    best_individual_seq = gaSeq.run()
    seq_time = time.time() - start_time


    start_time = time.time()
    gaPar = parallelGA.BackpackGAParralel(items, max_weight, population_size=200, generations=100, mutation_rate=0.1, verbose=False)
    best_individual_par = gaPar.run(12)
    par_time = time.time() - start_time

    start_time = time.time()
    gaMaster = masterSlaveGA.MasterSlaveBackpackGA(items, max_weight, population_size=20, generations=10, mutation_rate=0.1)
    best_individual_master = gaMaster.run(12)
    master_time = time.time() - start_time

    print("\n=== Результати порівняння ===")
    print(f"Послідовний алгоритм: {seq_time:.2f} сек")
    print(f"Паралельний алгоритм Island: {par_time:.2f} сек")
    print(f"Прискорення Island: {seq_time/par_time:.2f}x")
    print(f"Паралельний алгоритм Master-Slave: {master_time:.2f} сек")
    print(f"Прискорення Master-Slave: {seq_time/master_time:.2f}x")


