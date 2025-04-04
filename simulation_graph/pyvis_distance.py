from pyvis.network import Network
import networkx as nx

def make_distance_graph(G: nx.DiGraph):
    """Convert similarity weights to distance weights (1 - similarity)"""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if "weight" in data:
            H.add_edge(u, v, weight=1.0 - data["weight"])
    return H

def get_artist_node(G: nx.DiGraph, artist_name: str):
    """Find unique node for given artist name"""
    nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("name") == artist_name]
    if not nodes:
        raise ValueError(f"Artist '{artist_name}' not found in graph")
    if len(nodes) > 1:
        raise ValueError(f"Multiple nodes found for '{artist_name}'")
    return nodes[0][0]

GRAPH_PATH = "../similarity_graph.gexf"
ARTIST_A = "Metallica"
ARTIST_B = "BABYMETAL"
FONT_SIZE = 100

# Load and prepare graph
G = nx.read_gexf(GRAPH_PATH)
distance_graph = make_distance_graph(G)

# Find artist nodes
source_id = get_artist_node(G, ARTIST_A)
target_id = get_artist_node(G, ARTIST_B)

# Calculate shortest path
try:
    path = nx.shortest_path(distance_graph, source=source_id, target=target_id, weight="weight")
    path_length = nx.shortest_path_length(distance_graph, source=source_id, target=target_id, weight="weight")
except nx.NetworkXNoPath:
    raise ValueError(f"No path exists between {ARTIST_A} and {ARTIST_B}")

# Create subgraph containing only the path nodes
path_subgraph = G.subgraph(path)

# Add path sequence as an attribute for visualization
for i, node in enumerate(path):
    path_subgraph.nodes[node]["path_order"] = i

# Print path information
print(f"Shortest path from {ARTIST_A} to {ARTIST_B} (total distance: {path_length:.3f}):")
for node_id in path:
    artist_name = G.nodes[node_id].get("name", "Unknown")
    print(f"- {artist_name}")

net = Network(height=800, width=1920, notebook=False, filter_menu=True)
net.toggle_hide_edges_on_drag(False)
net.barnes_hut()
net.from_nx(path_subgraph)


for node in net.nodes:
    node['label'] = path_subgraph.nodes[node['id']].get('name', node['id'])
    node['font'] = {'size': FONT_SIZE}

net.show("graph.html", notebook=False)