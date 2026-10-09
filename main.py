"""
Лабораторная работа № 1. Вариант 8.
Генетический алгоритм для минимизации функции Захарова.

Запуск:
    python -m pip install numpy pandas matplotlib
    python main.py
    python main.py --runs 20 --budget 10000 --seed 42
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIMENSION = 7
LOWER_BOUND = -5.0
UPPER_BOUND = 10.0
DEFAULT_BUDGET = 10_000
DEFAULT_RUNS = 20
DEFAULT_SEED = 42


def zakharov(x):
    """Функция Захарова; возвращает одно числовое значение."""
    i = np.arange(1, len(x) + 1, dtype=float)
    weighted_sum = np.sum(0.5 * i * x)
    return float(np.sum(x ** 2) + weighted_sum ** 2 + weighted_sum ** 4)


def tournament_selection(population, fitness, rng, tournament_size=3):
    """Турнирный отбор: возвращает лучшую из случайно выбранных особей."""
    ids = rng.choice(len(population), size=tournament_size, replace=False)
    winner = ids[np.argmin(fitness[ids])]
    return population[winner].copy()


def genetic_algorithm(pop_size, seed, budget=DEFAULT_BUDGET,
                      crossover_probability=0.9,
                      mutation_probability=None, mutation_sigma=0.5):
    """
    Генетический алгоритм с точным бюджетом вызовов zakharov.
    Каждое значение в evaluations соответствует одному реальному вызову
    целевой функции. Значение элиты переносится вместе с кэшированным fitness.
    """
    if pop_size < 3:
        raise ValueError("Размер популяции должен быть не меньше 3.")
    if budget < pop_size:
        raise ValueError("Бюджет должен быть не меньше размера популяции.")
    if mutation_probability is None:
        mutation_probability = 1.0 / DIMENSION

    rng = np.random.default_rng(seed)
    population = rng.uniform(LOWER_BOUND, UPPER_BOUND,
                             size=(pop_size, DIMENSION))
    fitness = np.array([zakharov(ind) for ind in population], dtype=float)
    evaluations = pop_size

    best_idx = int(np.argmin(fitness))
    best_x = population[best_idx].copy()
    best_value = float(fitness[best_idx])
    history = [(evaluations, best_value)]

    while evaluations < budget:
        elite_idx = int(np.argmin(fitness))
        elite = population[elite_idx].copy()
        elite_fitness = float(fitness[elite_idx])

        # Оцениваем столько потомков, сколько позволяет бюджет.
        number_of_children = min(pop_size - 1, budget - evaluations)
        children = []

        for _ in range(number_of_children):
            parent1 = tournament_selection(population, fitness, rng)
            parent2 = tournament_selection(population, fitness, rng)

            # Арифметический кроссовер.
            if rng.random() < crossover_probability:
                alpha = rng.random(DIMENSION)
                child = alpha * parent1 + (1.0 - alpha) * parent2
            else:
                child = parent1.copy()

            # Независимая мутация каждой координаты.
            mask = rng.random(DIMENSION) < mutation_probability
            if np.any(mask):
                child[mask] += rng.normal(
                    0.0, mutation_sigma, size=int(mask.sum())
                )

            # Проекция в допустимую область.
            child = np.clip(child, LOWER_BOUND, UPPER_BOUND)
            children.append(child)

        children = np.asarray(children, dtype=float)
        child_fitness = np.array([zakharov(ind) for ind in children],
                                 dtype=float)
        evaluations += len(children)

        # Элита и её уже известное значение; лишнего вызова функции нет.
        population = np.vstack((elite, children))
        fitness = np.concatenate(([elite_fitness], child_fitness))

        current_idx = int(np.argmin(fitness))
        if fitness[current_idx] < best_value:
            best_value = float(fitness[current_idx])
            best_x = population[current_idx].copy()

        history.append((evaluations, best_value))

    assert evaluations == budget, (evaluations, budget)
    return best_x, best_value, history, evaluations


def random_search(seed, budget=DEFAULT_BUDGET, batch_size=50):
    """Базовый случайный поиск ровно с budget вычислениями функции."""
    rng = np.random.default_rng(seed)
    best_x = None
    best_value = float("inf")
    evaluations = 0
    history = []

    while evaluations < budget:
        n = min(batch_size, budget - evaluations)
        points = rng.uniform(LOWER_BOUND, UPPER_BOUND,
                             size=(n, DIMENSION))
        values = np.array([zakharov(point) for point in points])
        idx = int(np.argmin(values))
        if values[idx] < best_value:
            best_value = float(values[idx])
            best_x = points[idx].copy()
        evaluations += n
        history.append((evaluations, best_value))

    assert evaluations == budget, (evaluations, budget)
    return best_x, best_value, history, evaluations


def align_histories(histories, budget, step=50):
    """Сопоставляет траектории по числу оценок, не рисуя точки до инициализации."""
    grid = np.arange(step, budget + 1, step, dtype=int)
    aligned = []
    for history in histories:
        xs = np.array([point[0] for point in history], dtype=int)
        ys = np.array([point[1] for point in history], dtype=float)
        indices = np.searchsorted(xs, grid, side="right") - 1
        values = np.full(grid.shape, np.nan, dtype=float)
        valid = indices >= 0
        values[valid] = ys[indices[valid]]
        aligned.append(values)
    return grid, np.asarray(aligned)


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение генетического алгоритма и случайного поиска."
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.runs < 20:
        raise ValueError("По требованиям лабораторной задайте не менее 20 запусков.")
    if args.budget < 100:
        raise ValueError("Бюджет должен быть не меньше 100.")

    configs = {"GA_50": 50, "GA_100": 100}
    records = []
    histories = {}

    for name, pop_size in configs.items():
        histories[name] = []
        for run in range(args.runs):
            seed = args.seed + run
            x, value, history, evaluations = genetic_algorithm(
                pop_size=pop_size, seed=seed, budget=args.budget
            )
            histories[name].append(history)
            records.append({
                "algorithm": name, "run": run + 1, "seed": seed,
                "best_value": value, "evaluations": evaluations,
                "solution": json.dumps(x.tolist(), ensure_ascii=False)
            })
            print(f"{name}, запуск {run + 1:02d}: f(x)={value:.10g}; "
                  f"оценок={evaluations}")

    histories["Random"] = []
    for run in range(args.runs):
        seed = args.seed + run
        x, value, history, evaluations = random_search(
            seed=seed, budget=args.budget
        )
        histories["Random"].append(history)
        records.append({
            "algorithm": "Random", "run": run + 1, "seed": seed,
            "best_value": value, "evaluations": evaluations,
            "solution": json.dumps(x.tolist(), ensure_ascii=False)
        })
        print(f"Random, запуск {run + 1:02d}: f(x)={value:.10g}; "
              f"оценок={evaluations}")

    df = pd.DataFrame(records)
    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("algorithm")["best_value"].agg(
        min="min", mean="mean", median="median", std="std", max="max"
    )
    summary.to_csv("summary.csv", encoding="utf-8-sig")
    print("\\nСводная статистика:")
    print(summary.to_string(float_format=lambda v: f"{v:.8g}"))

    # Средняя траектория и межквартильный диапазон по всем запускам.
    plt.figure(figsize=(10, 6))
    for name, runs in histories.items():
        grid, values = align_histories(runs, args.budget, step=50)
        mean = np.nanmean(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        plt.plot(grid, mean, label=name)
        plt.fill_between(grid, q25, q75, alpha=0.16)
    plt.xlabel("Число вычислений целевой функции")
    plt.ylabel("Лучшее найденное значение f(x)")
    plt.title("Сходимость: среднее и межквартильный диапазон по запускам")
    plt.yscale("symlog", linthresh=1e-8)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=180)
    plt.close()

    best = df.loc[df["best_value"].idxmin()]
    print("\\nЛучший запуск:")
    print("Алгоритм:", best["algorithm"])
    print("Номер запуска:", int(best["run"]))
    print("f(x):", f"{best['best_value']:.12g}")
    print("x:", best["solution"])
    print("Проверка бюджета:", sorted(df["evaluations"].unique().tolist()))


if __name__ == "__main__":
    main()
