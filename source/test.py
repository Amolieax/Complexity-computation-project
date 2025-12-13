import numpy as np
import time
from tsp_solver import OneTwoTSP

class TSPTester:
    """Класс для тестирования алгоритма TSP"""
    
    def __init__(self):
        self.test_results = []
    
    def generate_random_test(self, n, test_name=''):
        """Генерирует случайный тест"""
        tsp = OneTwoTSP(n)
        matrix = tsp.generate_random_instance()

        return self._run_test(tsp, matrix, test_name)
    
    def run_custom_test(self, matrix, test_name=''):
        """Запускает тест с пользовательской матрицей"""

        n = matrix.shape[0]
        tsp = OneTwoTSP(n)
        tsp.load_from_matrix(matrix)
        
        return self._run_test(tsp, matrix, test_name)
    
    def run_specific_test(self, test_type, n, **kwargs):
        """Запускает специфический тип теста"""
        if test_type == "clustered":
            return self._generate_clustered_test(n, **kwargs)
        elif test_type == "sparse":
            return self._generate_sparse_test(n, **kwargs)
        elif test_type == "dense":
            return self._generate_dense_test(n, **kwargs)
        return None
    
    def _run_test(self, tsp, matrix, test_name):
        """Основная функция запуска теста"""
        start_time = time.time()
        tour, cost = tsp.berman_karpinski_algorithm()
        end_time = time.time()
        exec_time = end_time - start_time
        
        results = {
            'test_name': test_name,
            'n': tsp.n,
            'matrix': matrix.copy(),
            'tour': tour,
            'cost': cost,
            'time': exec_time,
            'tour_length': len(tour)
        }
        
        self._analyze_edges(results, tsp, tour)
        
        if tsp.n <= 10:
            self._find_optimal(results, tsp)
        
        self.test_results.append(results)
        self._print_results(results)
        
        return results
    
    def _analyze_edges(self, results, tsp, tour):
        """Анализирует ребра в туре"""
        weight_1_count = 0
        weight_2_count = 0
        
        for i in range(len(tour) - 1):
            weight = tsp.matrix[tour[i]][tour[i + 1]]
            if weight == 1:
                weight_1_count += 1
            else:
                weight_2_count += 1
        
        total_edges = len(tour) - 1
        
        results.update({
            'weight_1_count': weight_1_count,
            'weight_2_count': weight_2_count,
            'weight_1_ratio': weight_1_count / total_edges if total_edges > 0 else 0,
            'total_edges': total_edges
        })
    
    def _find_optimal(self, results, tsp):
        """Находит оптимальное решение для маленьких графов"""
        if tsp.n <= 8:
            optimal_tour, optimal_cost = tsp.exhaustive_search_optimal()
            if optimal_tour:
                results['optimal_cost'] = optimal_cost
                results['approx_ratio'] = results['cost'] / optimal_cost
                results['optimal_tour'] = optimal_tour
    
    def _print_results(self, results):
        """Выводит результаты теста"""
        print(f"Размер графа: {results['n']}")
        print(f"Время выполнения: {results['time']:.4f} секунд")
        print(f"Стоимость тура: {results['cost']}")
        print(f"Длина тура: {results['tour_length']} вершин")
        print(f"Ребра веса 1: {results['weight_1_count']} ({results['weight_1_ratio']:.1%})")
        print(f"Ребра веса 2: {results['weight_2_count']}")
        
        if 'optimal_cost' in results:
            print(f"Оптимальная стоимость: {results['optimal_cost']}")
            print(f"Коэффициент приближения: {results['approx_ratio']:.4f}")
        
        print(f"Тур: {results['tour'][:10]}{'...' if len(results['tour']) > 10 else ''}\n")
    
    def _generate_clustered_test(self, n, num_clusters=3):
        """Генерирует граф с кластерами"""
        matrix = np.ones((n, n), dtype=int) * 2
        np.fill_diagonal(matrix, 0)
        
        cluster_size = n // num_clusters
        for cluster in range(num_clusters):
            start = cluster * cluster_size
            end = min((cluster + 1) * cluster_size, n)
            
            for i in range(start, end):
                for j in range(i + 1, end):
                    if np.random.random() < 0.8:
                        matrix[i][j] = 1
                        matrix[j][i] = 1
        
        test_name = f"Кластеризованный граф {n} на {n} ({num_clusters} кластеров)"
        return self.run_custom_test(matrix, test_name)
    
    def _generate_sparse_test(self, n, density=0.2):
        """Генерирует разреженный граф"""
        matrix = np.ones((n, n), dtype=int) * 2
        np.fill_diagonal(matrix, 0)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < density:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        
        test_name = f"Разреженный граф {n} на {n} (density={density})"
        return self.run_custom_test(matrix, test_name)
    
    def _generate_dense_test(self, n, density=0.8):
        """Генерирует плотный граф"""
        matrix = np.ones((n, n), dtype=int) * 2
        np.fill_diagonal(matrix, 0)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < density:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        
        test_name = f"Плотный граф {n} на {n} (density={density})"
        return self.run_custom_test(matrix, test_name)
    
    def run_performance_test(self, sizes, num_tests_per_size=3):
        """Запускает тесты производительности для разных размеров"""
        print("\nТест производительности\n")

        performance_results = []
        
        for size in sizes:
            print(f"\nТестирование размера {size}:")
            size_results = []
            
            for i in range(num_tests_per_size):
                test_name = f"Тест {i + 1} для n = {size}"
                results = self.generate_random_test(size, test_name)
                size_results.append(results)
                
                print(f"готово (время: {results['time']:.3f}с)\n")

            avg_time = np.mean([r['time'] for r in size_results])
            avg_cost = np.mean([r['cost'] for r in size_results])
            avg_ratio = np.mean([r['weight_1_ratio'] for r in size_results])
            
            performance_results.append({
                'size': size,
                'avg_time': avg_time,
                'avg_cost': avg_cost,
                'avg_ratio': avg_ratio,
                'tests': size_results,
                'std_time': np.std([r['time'] for r in size_results]),
                'std_cost': np.std([r['cost'] for r in size_results])
            })
        
        return performance_results
    
    def get_all_results(self):
        """Возвращает все результаты тестов"""
        return self.test_results

    def clear_results(self):
        """Очищает результаты тестов"""
        self.test_results = []