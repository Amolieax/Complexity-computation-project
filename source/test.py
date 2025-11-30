from one_two_tsp import OneTwoTSP
from itertools import combinations
import time
import numpy as np


class Test:
    def __init__(self, label):
        self.label = label

    def set_matrix(self, size=None, matrix=None):
        """Устанавливает/генерирует матрицу для теста"""
        if matrix is None:
            self.tsp = OneTwoTSP(size)
            self.matrix = self.tsp.generate_random_instance()
        else:
            self.matrix = matrix
            self.tsp = OneTwoTSP(matrix.shape[0])
            self.tsp.load_from_matrix(matrix)

    def test(self):
        print(self.label)
        print("Матрица расстояний:")
        print(self.matrix)
        print()
        
        start_time = time.time()
        berman_tour, berman_cost, christofides_cost, improved_cost = self.tsp.berman_karpinski_algorithm()
        end_time = time.time()
        
        print(f"Алгоритм Кристофидеса: {christofides_cost}")
        print(f"После локального поиска: {improved_cost}")
        print(f"Алгоритм Бермана-Карпински: {berman_cost}")
        print(f"Тур: {berman_tour}")
        print(f"Время выполнения: {end_time - start_time:.4f} секунд")
        
        if self.tsp.n <= 8:
            optimal_tour, optimal_cost = self.tsp.exhaustive_search_optimal()
            if optimal_tour:
                print(f"Точное решение: {optimal_cost}")
                print(f"Коэффициент приближения: {berman_cost / optimal_cost:.4f} / {self.tsp.approx_const():.4f}")

        print("\nАнализ качества:")
        print(f"Улучшение относительно Кристофидеса: {((christofides_cost - berman_cost) / christofides_cost * 100):.2f}%")
        
        edge_weights = []
        for i in range(len(berman_tour) - 1):
            weight = self.matrix[berman_tour[i]][berman_tour[i + 1]]
            edge_weights.append(weight)
        
        weight_1_count = edge_weights.count(1)
        weight_2_count = edge_weights.count(2)
        
        print(f"Ребра стоимости 1 в туре: {weight_1_count}")
        print(f"Ребра стоимости 2 в туре: {weight_2_count}")
        print(f"Доля ребер стоимости 1: {weight_1_count / len(edge_weights):.2f}")

        print("\n" + "="*50 + "\n")

def test_algorithm():
    """Тестирование алгоритма на различных экземплярах"""
    
    print("=== Тестирование алгоритма Бермана-Карпински для (1,2)-TSP ===\n")
    
    test1 = Test("Тест 1: Случайная матрица 8X8")
    test1.set_matrix(size=8)
    test1.test()
        
    test2 = Test("Тест 2: Специальный экземпляр с известной структурой")
    special_matrix = np.array([
        [0, 1, 2, 2, 2, 2, 1, 2],
        [1, 0, 2, 2, 2, 2, 2, 1],
        [2, 2, 0, 1, 2, 2, 2, 2],
        [2, 2, 1, 0, 2, 2, 2, 2],
        [2, 2, 2, 2, 0, 1, 2, 2],
        [2, 2, 2, 2, 1, 0, 2, 2],
        [1, 2, 2, 2, 2, 2, 0, 1],
        [2, 1, 2, 2, 2, 2, 1, 0]
    ])
    test2.set_matrix(matrix=special_matrix)
    test2.test()

    test3 = Test("Тест 3: Случайная матрица 15x15")
    test3.set_matrix(size=15)
    test3.test()


def performance_analysis():
    """Анализ производительности на экземплярах разного размера"""
    print("\n=== Анализ производительности ===")
    
    sizes = [5, 8, 10, 12, 15]
    times = []
    costs = []
    
    for size in sizes:
        tsp = OneTwoTSP(size)
        tsp.generate_random_instance()
        
        start_time = time.time()
        tour, cost, _, _ = tsp.berman_karpinski_algorithm()
        end_time = time.time()
        
        times.append(end_time - start_time)
        costs.append(cost)
        
        print(f"n={size}: время={end_time - start_time:.4f}с, стоимость={cost}")
    
    print("\nРост времени выполнения:")
    for i in range(1, len(sizes)):
        growth = times[i] / times[i-1] if times[i-1] > 0 else 0
        print(f"От {sizes[i-1]} до {sizes[i]}: в {growth:.2f} раз")


test_algorithm()
performance_analysis()