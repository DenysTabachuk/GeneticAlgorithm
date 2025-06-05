import random
from typing import List, Tuple, Type
from BackpackGA import BackpackGA


def test_against_known_optimum(
    algorithm_class: Type[BackpackGA],
    runs: int = 10,
    population_size: int = 20,
    generations: int = 100,
    mutation_rate: float = 0.1,
    items: List[Tuple[int, int]] = [
        (3, 1),   # 1
        (4, 5),   # 2
        (2, 3),   # 3
        (3, 7),   # 4 
        (5, 6),   # 5
        (1, 2),   # 6
        (6, 9)    # 7
    ],
    max_weight: int = 10,
    expected_value: int = 18
) -> None:
    print("=" * 60)
    print("ТЕСТ НА ВІДОМОМУ ОПТИМУМІ".center(60))
    print("=" * 60)
    print("Предмети:")
    for i, (w, v) in enumerate(items):
        print(f"  {i+1:>2}) вага = {w:<2}, цінність = {v}")
    print(f"\nМаксимальна вага рюкзака: {max_weight}")
    print(f"Очікувана цінність оптимального розв'язку: {expected_value}")
    print("\nПараметри алгоритму:")
    print(f"  - population_size = {population_size}")
    print(f"  - generations     = {generations}")
    print(f"  - mutation_rate   = {mutation_rate}")
    print(f"  - кількість запусків = {runs}")
    print("-" * 60)

    success_count = 0
    for i in range(runs):
        ga: BackpackGA = algorithm_class(
            items=items,
            max_weight=max_weight,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            verbose=False
        )

        best_solution: List[int]
        value: int
        weight: int
        best_solution, value, weight = ga.run()

        status = "УСПІХ" if value == expected_value else "НЕВДАЧА"
        if value == expected_value:
            success_count += 1

        print(f"Тест {i+1:>2}: {status:<7} | Цінність = {value:>2}, Вага = {weight:>2}, Розв'язок = {best_solution}")

    print("-" * 60)
    print(f"РЕЗУЛЬТАТ: {success_count} з {runs} тестів успішно ({(success_count / runs) * 100:.1f}%)")
    print("=" * 60)


# Параметри
NUM_ITEMS: int = 100

# Генерація випадкових предметів (вага, цінність)
items: List[Tuple[int, int]] = [
    (random.randint(1, 20), random.randint(10, 100)) for _ in range(NUM_ITEMS)
]

# Запуск тесту
test_against_known_optimum(algorithm_class=BackpackGA)
