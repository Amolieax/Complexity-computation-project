import numpy as np
import itertools
from berman_karpinski import BermanKarpinskiTSP


class OneTwoTSP:
    """Интерфейс для решения (1,2)-задачи коммивояжёра."""
    
    def __init__(self, n):
        self.n = n
        self.matrix = None
        self.approx_const_val = 8/7
        self.algo = BermanKarpinskiTSP(n)
    
    def approx_const(self):
        return self.approx_const_val
    
    def load_from_matrix(self, matrix):
        self.matrix = matrix.copy()
        self.n = matrix.shape[0]
        self.algo = BermanKarpinskiTSP(self.n)
        self.algo.load_from_matrix(matrix)
    
    def generate_random_instance(self):
        matrix = np.ones((self.n, self.n), dtype=int) * 2
        np.fill_diagonal(matrix, 0)
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                if np.random.random() < 0.4:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        
        self.matrix = matrix
        self.algo.load_from_matrix(matrix)
        return matrix
    
    def tour_cost(self, tour):
        if not tour or len(tour) < 2:
            return 0
        cost = 0
        for i in range(len(tour)-1):
            cost += self.matrix[tour[i]][tour[i+1]]
        if tour[0] != tour[-1]:
            cost += self.matrix[tour[-1]][tour[0]]
        return cost
    
    def exhaustive_search_optimal(self):
        if self.n > 8:
            return None, float('inf')
        
        best_tour = None
        best_cost = float('inf')
        
        for perm in itertools.permutations(range(1, self.n)):
            tour = [0] + list(perm) + [0]
            cost = self.tour_cost(tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        return best_tour, best_cost
    
    def berman_karpinski_algorithm(self):
        tour, cost = self.algo.berman_karpinski_algorithm()
        return tour, cost