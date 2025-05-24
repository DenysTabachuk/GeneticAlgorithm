import random
import time
from TSP_GA import TSP_GA
from MasterSlaveTSP_GA import MasterSlaveTSP_GA  # виправив назву з MasterSalveGa на MasterSlaveTSP_GA

def main():
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(300)]

    print("=== Послідовний TSP_GA ===")
    ga_seq = TSP_GA(cities, generations=200, population_size=100, verbose=False)
    start = time.time()
    best_path_seq, best_dist_seq = ga_seq.run()
    end = time.time()
    print(f"Найкоротший шлях: {best_path_seq}")
    print(f"Довжина шляху: {best_dist_seq:.2f}")
    print(f"Час виконання: {end - start:.2f} секунд\n")

    print("=== Паралельний MasterSlaveTSP_GA ===")
    ga_par = MasterSlaveTSP_GA(cities, generations=200, population_size=100, verbose=False)
    start = time.time()
    best_path_par, best_dist_par = ga_par.run()
    end = time.time()
    print(f"Найкоротший шлях: {best_path_par}")
    print(f"Довжина шляху: {best_dist_par:.2f}")
    print(f"Час виконання: {end - start:.2f} секунд\n")

    # Порівняння результатів
    print("=== Порівняння ===")
    print(f"Різниця в довжині шляху: {best_dist_seq - best_dist_par:.2f}")
    print(f"Різниця у часі: {(end - start):.2f} секунд (паралельний - послідовний)")

if __name__ == "__main__":
    main()
