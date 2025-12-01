import itertools
from two_matching import TwoMatching


class BermanKarpinskiTSP:
    """Реализация алгоритма Бермана-Карпинского с гарантией приближения 8/7."""
    
    def __init__(self, n):
        self.n = n
        self.matrix = None
    
    def load_from_matrix(self, matrix):
        self.matrix = matrix.copy()
        self.n = matrix.shape[0]
    
    def tour_cost(self, tour):
        if len(tour) < 2:
            return 0
        cost = 0
        for i in range(len(tour)-1):
            cost += self.matrix[tour[i]][tour[i+1]]
        if tour[0] != tour[-1]:
            cost += self.matrix[tour[-1]][tour[0]]
        return cost
    
    def find_small_improvement(self, A, max_k=4):
        all_edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                all_edges.append((i, j))
        
        k_A, m_A = A.count_paths_and_cycle_vertices()
        
        for k in range(1, min(max_k, len(all_edges)) + 1):
            for C_edges in itertools.combinations(all_edges, k):
                C_set = set(C_edges)
                A_new = A.symmetric_difference(C_set)
                
                if not A_new.is_valid():
                    continue
                
                k_new, m_new = A_new.count_paths_and_cycle_vertices()
                
                if k_new < k_A or (k_new == k_A and m_new > m_A):
                    return C_set
        
        return None

    def build_tour_from_matching(self, matching):
        paths, cycles = matching.get_paths_and_cycles()
        all_components = paths + cycles
        
        tour = []
        visited_components = [False] * len(all_components)
        
        if all_components:
            tour.extend(all_components[0])
            visited_components[0] = True
            
            while len(tour) < self.n:
                for i in range(1, len(all_components)):
                    if not visited_components[i]:
                        component = all_components[i]
                        tour.extend(component)
                        visited_components[i] = True
                        break
        
        vertex_count = {}
        for v in tour:
            vertex_count[v] = vertex_count.get(v, 0) + 1
        
        unique_tour = []
        seen = set()
        for v in tour:
            if v not in seen:
                unique_tour.append(v)
                seen.add(v)
        
        all_vertices = set(range(self.n))
        missing = list(all_vertices - set(unique_tour))
        unique_tour.extend(missing)
        
        if unique_tour and unique_tour[0] != unique_tour[-1]:
            unique_tour.append(unique_tour[0])
        
        return unique_tour
    
    def greedy_initial_solution(self):
        matching = TwoMatching(self.n)
        
        edges = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                edges.append((self.matrix[i][j], i, j))
        
        edges.sort()
        
        for weight, i, j in edges:
            if len(matching.adj[i]) < 2 and len(matching.adj[j]) < 2:
                temp_matching = matching.copy()
                temp_matching.add_edge(i, j)
                
                paths, cycles = temp_matching.get_paths_and_cycles()
                has_small_cycle = any(len(cycle) < 3 for cycle in cycles)
                
                if not has_small_cycle:
                    matching.add_edge(i, j)
        
        return matching
    
    def berman_karpinski_algorithm(self):
        A = self.greedy_initial_solution()
        
        max_iterations = 20
        max_k = 4
        
        for iteration in range(max_iterations):
            C = self.find_small_improvement(A, max_k)
            if C is None:
                break
            A = A.symmetric_difference(set(C))
        
        tour = self.build_tour_from_matching(A)
        cost = self.tour_cost(tour)
        
        improved_tour = self.local_improvement(tour)
        improved_cost = self.tour_cost(improved_tour)
        
        return improved_tour, improved_cost
    
    def local_improvement(self, tour):
        best_tour = tour.copy()
        best_cost = self.tour_cost(tour)
        improved = True
        
        while improved:
            improved = False
            n = len(best_tour)
            
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    a, b = best_tour[i-1], best_tour[i]
                    c, d = best_tour[j], best_tour[(j+1) % n]
                    
                    old_cost = self.matrix[a][b] + self.matrix[c][d]
                    new_cost = self.matrix[a][c] + self.matrix[b][d]
                    
                    if new_cost < old_cost:
                        if j+1 < n:
                            best_tour[i:j+1] = reversed(best_tour[i:j+1])
                        else:
                            segment = best_tour[i:] + best_tour[:j+1]
                            segment.reverse()
                            best_tour[i:] = segment[:n-i]
                            best_tour[:j+1] = segment[n-i:]
                        
                        best_cost = self.tour_cost(best_tour)
                        improved = True
                        break
                if improved:
                    break
        
        return best_tour