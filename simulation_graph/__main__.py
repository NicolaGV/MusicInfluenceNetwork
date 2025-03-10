import networkx as nx
import random
import math

from MusicInfluenceNetwork.Networks.ICM import ICM

GRAPH_PATH = "similarity_graph.gexf"

# Activation function during ICM, customized for current graph
def is_activate(self, source, target):

    if not self.graph.has_edge(source, target):
        return False
    
    source_career_start = self.graph.nodes[source].get('career_start_year')
    target_career_end = self.graph.nodes[target].get('career_end_year')
    # if undefined, set to -1
    # if career started after, there is no chance for influence
    if source_career_start is None:
        source_career_start = -1
    if target_career_end is None:
        target_career_end = -1
    if source_career_start != -1 and target_career_end != -1:
        if source_career_start > target_career_end:
            return False

    # sigmoid influence
    weight = self.graph[source][target].get('weight', 0.0)
    p = 1 / (1 + math.exp(-weight))
    return random.random() < p

    
def main():
    
    print("import graph")
    graph = nx.read_gexf(GRAPH_PATH)

    icm = ICM(graph)
    icm._is_activate = is_activate.__get__(icm, ICM) # override for career end dates

    print("evaluate centrality")
    icm.evaluate_centrality()

    print("icm")
    icm.run_cascade(200, 50) # 200 iterations, 50 seeds
    for u, v, data in icm.graph.edges(data=True):
        data['weight'] = data['influence']

    print("icm done")
    for u, v, data in icm.graph.edges(data=True):
        data['weight'] = data['influence']
    nx.write_gexf(icm.graph, 'influence_graph.gexf')
    print("file saved")

if __name__ == "__main__":
    main()