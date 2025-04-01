

from pyvis.network import Network
import networkx as nx

def make_distance_graph(G: nx.DiGraph):
    # Create a new graph with transformed edge weights: distance = 1 - similarity
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))  # Preserve all nodes and their attributes

    for u, v, data in G.edges(data=True):
        original_weight = data.get("weight", None)
        if original_weight is not None:
            H.add_edge(u, v, weight=1.0 - original_weight) 
    return H

GRAPH_PATH = "similarity_graph.gexf"

ARTIST = "BABYMETAL"

MAX_DISTANCE = 0.99

G: nx.DiGraph = nx.read_gexf(GRAPH_PATH)

nodes = list(G.nodes(data=True))

artist_nodes = [node for node in nodes if node[1]["name"] == ARTIST]

if not artist_nodes:
    raise ValueError(f"Artist '{ARTIST}' not found in the graph.")
if len(artist_nodes) > 1:
    raise ValueError(f"Multiple nodes found for artist '{ARTIST}'.")

source_node_id = artist_nodes[0][0]

distance_graph = make_distance_graph(G)

shortest_paths = nx.shortest_path_length(distance_graph, source=source_node_id, weight="weight")

pruned_graph = nx.DiGraph()

# Filter nodes by distance threshold
filtered_artists = {
    G.nodes[node_id].get("name", "Unknown"): distance
    for node_id, distance in shortest_paths.items()
    if distance < MAX_DISTANCE  # Keep only nodes below the threshold
}

# Get node IDs of artists within the distance threshold
kept_node_ids = [
    node_id 
    for node_id, distance in shortest_paths.items() 
    if distance < MAX_DISTANCE
]

# Create a subgraph from the ORIGINAL graph (G) with similarity weights
subgraph = G.subgraph(kept_node_ids)

# Save the subgraph to a new file
output_path = f"{ARTIST}_subgraph.gexf"
nx.write_gexf(subgraph, output_path)
print(f"Subgraph saved to {output_path}")

print(f"Artists within distance {MAX_DISTANCE} from {ARTIST}:")
for name, dist in sorted(filtered_artists.items(), key=lambda x: x[1]):
    print(f"- {name}: {dist:.3f}")

net = Network(height=800, width=1920, notebook=False, filter_menu=True)
net.toggle_hide_edges_on_drag(False)
net.barnes_hut()
net.from_nx(subgraph)
net.show("graph.html", notebook=False)