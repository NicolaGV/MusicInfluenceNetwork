import networkx as nx
import random
import math

class ICM:

    def __init__(self, graph):
        self.graph = graph
        for u, v in self.graph.edges():
            self.graph[u][v]['influence'] = 0
    
    def evaluate_centrality(self):
        eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
        nx.set_node_attributes(self.graph, eigenvector_centrality, 'eigenvector_centrality')

    def _choose_seeds(self, num_seeds):
        # sorted_nodes = sorted(self.graph.nodes, key=lambda n: self.graph.nodes[n]['eigenvector_centrality'], reverse=True)

        centrality_scores = nx.get_node_attributes(self.graph, 'eigenvector_centrality')
        total_centrality = sum(centrality_scores.values())
        probabilities = {node: score / total_centrality for node, score in centrality_scores.items()}
        
        seeds = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=num_seeds
        )
        return seeds

    def _is_activate(self, source, target):
        if self.graph.has_edge(source, target):
            weight = self.graph[source][target].get('weight', 0.0)
            p = 1 / (1 + math.exp(-weight))
            return random.random() < p
        else:
            return False

    
    def run_cascade(self, num_iterations, num_seeds):
        for i in range(num_iterations):
            print(f"Iteration {i + 1}", end='\r')
            seed_nodes = self._choose_seeds(num_seeds)
            self._run_cascade_iteration(seed_nodes)
        
        for u, v in self.graph.edges():
            self.graph[u][v]['influence'] /= num_iterations
    
    def _run_cascade_iteration(self, seed_nodes):

        active_nodes = set(seed_nodes)
        new_active_nodes = set(seed_nodes)
        
        while new_active_nodes:
            next_active_nodes = set()
            for node in new_active_nodes:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in active_nodes and self._is_activate(node, neighbor):
                        next_active_nodes.add(neighbor)
                        self.graph[node][neighbor]['influence'] += 1
            active_nodes.update(next_active_nodes)
            new_active_nodes = next_active_nodes
        
        return active_nodes

# # Use case test:

# num_nodes = 10000
# large_graph = nx.watts_strogatz_graph(num_nodes, k=10, p=0.1)
# large_graph = large_graph.to_directed()

# icm = ICM(graph=large_graph)
# icm.evaluate_centrality()
# seeds = icm.choose_seeds(num_seeds=10)
# icm.run_cascade(10000, seeds)
# for u, v, data in icm.graph.edges(data=True):
#     data['weight'] = data['influence']
# nx.write_gexf(icm.graph, 'graph_with_influence_10000.gexf')