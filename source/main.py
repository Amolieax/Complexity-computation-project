import numpy as np
from test import TSPTester
from visualize import TSPVisualizer

def main():
    tester = TSPTester()
    visualizer = TSPVisualizer()
    
    tester.generate_random_test(8, "Случайный граф 8 на 8")
    
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
    tester.run_custom_test(special_matrix, "Специальная матрица 8 на 8")

    tester.generate_random_test(8, "Случайный граф 8 на 8 2")

    tester.generate_random_test(8, "Случайный граф 8 на 8 3")

    tester.generate_random_test(8, "Случайный граф 8 на 8 4")

    tester.generate_random_test(8, "Случайный граф 8 на 8 5")
    
    tester.generate_random_test(12, "Случайный граф 12 на 12")
    
    tester.run_specific_test("clustered", 10, num_clusters=3)
    
    tester.run_specific_test("sparse", 10, density=0.3)
    
    all_results = tester.get_all_results()
    
    visualizer.plot_comparative_analysis(all_results, "comparative_analysis.png")
    visualizer.plot_edge_distribution(all_results, "edge_distribution.png")
    
    sizes = [5, 8, 10, 12, 15]
    performance_results = tester.run_performance_test(sizes, num_tests_per_size=2)    
    visualizer.plot_performance_analysis(performance_results, "performance_analysis.png")
    
    visualizer.save_all_figures()
    visualizer.show_all_figures()

    if all_results:
        avg_time = np.mean([r['time'] for r in all_results])
        avg_ratio = np.mean([r['weight_1_ratio'] for r in all_results])

        print(f"Среднее время выполнения: {avg_time:.3f} секунд")
        print(f"Средняя доля ребер веса 1: {avg_ratio:.1%}")

        optimal_tests = [r for r in all_results if 'approx_ratio' in r]
        if optimal_tests:
            avg_approx = np.mean([r['approx_ratio'] for r in optimal_tests])
            max_approx = max([r['approx_ratio'] for r in optimal_tests])
            print(f"Средний коэффициент приближения: {avg_approx:.4f}")
            print(f"Максимальный коэффициент приближения: {max_approx:.4f}")
            print(f"Теоретическая граница: 1.1429")

if __name__ == "__main__":
    main()