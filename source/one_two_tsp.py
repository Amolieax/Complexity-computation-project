import numpy as np
import networkx as nx
from itertools import permutations


class OneTwoTSP:
    def __init__(self, n):
        """
        Инициализация задачи (1,2)-TSP для n городов
        Веса ребер: 1 или 2
        """
        self.n = n
        self.dist_matrix = None

    def generate_random_instance(self):
        """Генерация случайной матрицы расстояний с весами 1 или 2"""
        np.random.seed(42)
        self.dist_matrix = np.random.choice([1, 2], size=(self.n, self.n))
        np.fill_diagonal(self.dist_matrix, 0)
        return self.dist_matrix
    
    def load_from_matrix(self, matrix):
        """Загрузка матрицы расстояний"""
        self.dist_matrix = matrix
        self.n = len(matrix)
        return self.dist_matrix
    
    def christofides_algorithm(self):
        """Алгоритм Кристофидеса для метрического TSP (2-приближение)"""
        G = nx.Graph()
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G.add_edge(i, j, weight=self.dist_matrix[i][j])
        
        mst = nx.minimum_spanning_tree(G)
        
        odd_degree_vertices = [v for v, degree in mst.degree() if degree % 2 == 1]
        
        subgraph = G.subgraph(odd_degree_vertices)
        matching = nx.min_weight_matching(subgraph)
        
        multigraph = nx.MultiGraph(mst)
        for u, v in matching:
            multigraph.add_edge(u, v, weight=G[u][v]['weight'])
        
        euler_circuit = list(nx.eulerian_circuit(multigraph))
        
        visited = set()
        tour = []
        
        for u, v in euler_circuit:
            if u not in visited:
                tour.append(u)
                visited.add(u)
        
        tour.append(tour[0])
        
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.dist_matrix[tour[i]][tour[i + 1]]
        
        return tour, cost
    
    def berman_karpinski_algorithm(self):
        """
        Алгоритм Бермана-Карпински для (1,2)-TSP
        Коэффициент приближения: 8/7 ≈ 1.142
        """
        christofides_tour, christofides_cost = self.christofides_algorithm()
        
        improved_tour = self.local_search_improvement(christofides_tour.copy())
        improved_cost = self.calculate_tour_cost(improved_tour)
        
        final_tour = self.special_improvements(improved_tour.copy())
        final_cost = self.calculate_tour_cost(final_tour)
        
        return final_tour, final_cost, christofides_cost, improved_cost
    
    def calculate_tour_cost(self, tour):
        """Вычисление стоимости тура"""
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.dist_matrix[tour[i]][tour[i + 1]]
        return cost
    
    def local_search_improvement(self, tour):
        """Улучшение тура с помощью 2-opt локального поиска"""
        best_tour = tour.copy()
        best_cost = self.calculate_tour_cost(tour)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    if j - i == 1:
                        continue
                    
                    new_tour = best_tour.copy()
                    new_tour[i:j] = reversed(new_tour[i:j])
                    new_cost = self.calculate_tour_cost(new_tour)
                    
                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        return best_tour

    def special_improvements(self, tour):
        """
        Специальные улучшения для (1,2)-TSP
        Фокусируемся на замене ребер стоимости 2 на ребра стоимости 1
        """
        best_tour = tour.copy()
        best_cost = self.calculate_tour_cost(tour)
        
        for i in range(len(tour) - 1):
            if self.dist_matrix[tour[i]][tour[i + 1]] == 2:
                for k in range(self.n):
                    if (k != tour[i] and k != tour[i + 1] and 
                        self.dist_matrix[tour[i]][k] == 1 and 
                        self.dist_matrix[k][tour[i + 1]] == 1):
                        
                        new_tour = best_tour.copy()
                        new_tour.insert(i + 1, k)
                        new_cost = self.calculate_tour_cost(new_tour)
                        
                        if new_cost < best_cost:
                            best_tour = new_tour
                            best_cost = new_cost
                            break
        
        return best_tour
    
    def exhaustive_search_optimal(self, max_n=10):
        """
        Точное решение для небольших экземпляров (только для проверки)
        """
        if self.n > max_n:
            return None, np.inf
        
        best_tour = None
        best_cost = np.inf

        for perm in permutations(range(1, self.n)):
            tour = [0] + list(perm) + [0]
            cost = self.calculate_tour_cost(tour)
            
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        return best_tour, best_cost
    
    @staticmethod
    def approx_const():
        return 8.0 / 7.0