class TwoMatching:
    """
    2-совпадение - множество рёбер, где степень каждой вершины не превышает 2.
    Соответствует определению из статьи Бермана-Карпинского.
    """
    
    def __init__(self, n):
        self.n = n
        self.edges = set()
        self.adj = [[] for _ in range(n)]
            
    def add_edge(self, u, v):
        if u == v:
            return
        edge = (min(u, v), max(u, v))
        if edge not in self.edges:
            self.edges.add(edge)
            self.adj[u].append(v)
            self.adj[v].append(u)
                
    def remove_edge(self, u, v):
        edge = (min(u, v), max(u, v))
        if edge in self.edges:
            self.edges.remove(edge)
            self.adj[u].remove(v)
            self.adj[v].remove(u)
                
    def copy(self):
        new_matching = TwoMatching(self.n)
        for u, v in self.edges:
            new_matching.add_edge(u, v)
        return new_matching
        
    def symmetric_difference(self, other_edges):
        result = TwoMatching(self.n)
        all_edges = self.edges.union(other_edges)
        
        for edge in all_edges:
            in_A = edge in self.edges
            in_C = edge in other_edges
            if in_A != in_C:
                result.add_edge(edge[0], edge[1])
            
        return result
        
    def is_valid(self):
        for i in range(self.n):
            if len(self.adj[i]) > 2:
                return False
        return True
        
    def get_paths_and_cycles(self):
        visited = [False] * self.n
        paths = []
        cycles = []
            
        for start in range(self.n):
            if not visited[start] and len(self.adj[start]) == 1:
                path = []
                current = start
                prev = -1
                    
                while current != -1:
                    visited[current] = True
                    path.append(current)
                    next_v = -1
                    for neighbor in self.adj[current]:
                        if neighbor != prev and not visited[neighbor]:
                            next_v = neighbor
                            break
                    prev = current
                    current = next_v
                    
                if len(path) >= 2:
                    paths.append(path)
            
        for start in range(self.n):
            if not visited[start] and len(self.adj[start]) == 2:
                cycle = []
                current = start
                    
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    for neighbor in self.adj[current]:
                        if not visited[neighbor]:
                            current = neighbor
                            break
                    
                if len(cycle) >= 3:
                    cycles.append(cycle)
            
        for v in range(self.n):
            if not visited[v]:
                paths.append([v])
            
        return paths, cycles
        
    def count_paths_and_cycle_vertices(self):
        paths, cycles = self.get_paths_and_cycles()
        k = len(paths)
        m = sum(len(cycle) for cycle in cycles)
        return k, m