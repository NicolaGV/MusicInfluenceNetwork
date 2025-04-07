# app.py
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import uuid
import os
import networkx as nx
from pyvis.network import Network
import time
from tqdm import tqdm

# gat imports
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMILARITY_GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "similarity_graph.gexf")
INFLUENCE_GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "influence_graph.gexf")
ATTENTION_GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "attention_composite.gexf")
DISTANCE_GRAPH_PATH = os.path.join(os.path.dirname(SIMILARITY_GRAPH_PATH), "distance_graph.gexf")

FONT_SIZE = 25
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
LIB_DIR = os.path.join(APP_DIR, "lib")
MAX_NODES = 200


app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Precomputed graphs
DISTANCE_GRAPH = None

INFLUENCE_GRAPH = None
ATTENTION_GRAPH = None
SIMILARITY_GRAPH = None

graphs_loaded = False
similarity_loaded = False
influence_loaded = False
attention_loaded = False

# gat
GAT_MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "..", "gat_model.pth")
gat_model = None
gat_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimilarityGAT(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=64, heads=4, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.conv1 = GATConv(
            embedding_dim, 
            hidden_dim, 
            heads=heads,
            edge_dim=1,
            add_self_loops=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim * heads)
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            edge_dim=1,
            add_self_loops=True
        )
        
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.conv3 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            edge_dim=1,
            concat=False,
            add_self_loops=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = dropout

    def forward(self, data, return_attention=True):
        edge_attr = data.edge_attr.to(torch.float32).unsqueeze(-1)
        
        x = self.norm1(self.embedding(torch.arange(data.num_nodes, device=data.edge_index.device)))
        
        # Layer 1
        x, (attn_idx1, attn_weights1) = self.conv1(x, data.edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.leaky_relu(self.norm2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x, (attn_idx2, attn_weights2) = self.conv2(x, data.edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.leaky_relu(self.norm3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x, (attn_idx3, attn_weights3) = self.conv3(x, data.edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Decoder
        src, dst = data.edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        pred_scores = self.decoder(edge_features).squeeze()
        
        if return_attention:
            return pred_scores, (attn_weights1, attn_weights2, attn_weights3)
        return pred_scores

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def load_graphs():
    global graphs_loaded, SIMILARITY_GRAPH, DISTANCE_GRAPH
    if not graphs_loaded:
        SIMILARITY_GRAPH = nx.read_gexf(SIMILARITY_GRAPH_PATH)

        # Load or generate distance graph
        if os.path.exists(DISTANCE_GRAPH_PATH):
            print("Loading precomputed distance graph...")
            DISTANCE_GRAPH = nx.read_gexf(DISTANCE_GRAPH_PATH)
        else:
            print("Generating and saving distance graph...", flush=True)
            DISTANCE_GRAPH = nx.DiGraph()
            DISTANCE_GRAPH.add_nodes_from(SIMILARITY_GRAPH.nodes(data=True))

            total_edges = len(SIMILARITY_GRAPH.edges())
            processed_edges = 0
            print(f"Total edges to process: {total_edges}")

            for u, v, data in SIMILARITY_GRAPH.edges(data=True):
                if "weight" in data:
                    DISTANCE_GRAPH.add_edge(u, v, weight=1.0 - data["weight"])
                processed_edges += 1

                if processed_edges % 100000 == 0:
                    print(f"Processed {processed_edges}/{total_edges} edges", flush=True)
            nx.write_gexf(DISTANCE_GRAPH, DISTANCE_GRAPH_PATH)

def load_all_graphs():
    print("Loading all graphs...", flush=True)
    # load_influence_graph()
    # load_attention_graph()
    load_similarity_graph()
    load_gat_model()


def load_influence_graph():
    global influence_loaded, INFLUENCE_GRAPH
    if not influence_loaded:
        print("Loading influence graph...", flush=True)
        influence_loaded = True
        INFLUENCE_GRAPH = nx.read_gexf(INFLUENCE_GRAPH_PATH)

def load_attention_graph():
    global attention_loaded, ATTENTION_GRAPH
    if not attention_loaded:
        print("Loading attention graph...", flush=True)
        attention_loaded = True
        ATTENTION_GRAPH = nx.read_gexf(ATTENTION_GRAPH_PATH)

def load_similarity_graph():
    global similarity_loaded, SIMILARITY_GRAPH, node_id_to_idx
    if not similarity_loaded:
        print("Loading similarity graph...", flush=True)
        SIMILARITY_GRAPH = nx.read_gexf(SIMILARITY_GRAPH_PATH)
        
        node_id_to_idx = {
            data.get("name", node): idx 
            for idx, (node, data) in enumerate(SIMILARITY_GRAPH.nodes(data=True))
        }        
        similarity_loaded = True

def load_gat_model():
    global gat_model
    if gat_model is None:
        print("Loading GAT model...", flush=True)
        num_nodes = len(node_id_to_idx)
        gat_model = SimilarityGAT(num_nodes=num_nodes, embedding_dim=64, hidden_dim=64, heads=4, dropout=0.5)
        checkpoint = torch.load(GAT_MODEL_PATH, map_location=gat_device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        gat_model.load_state_dict(state_dict)
        gat_model.to(gat_device)
        gat_model.eval()
        print("GAT model loaded", flush=True)




def get_artist_node(G: nx.DiGraph, artist_name: str):
    nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("name") == artist_name]
    if not nodes:
        raise ValueError(f"Artist '{artist_name}' not found in graph")
    if len(nodes) > 1:
        raise ValueError(f"Multiple nodes found for '{artist_name}'")
    return nodes[0][0]

@app.route('/similarity-base', methods=['POST'])
def similarity_base():
    print("Get base similarity", flush=True)
    load_similarity_graph()
    data = request.get_json()
    artist_name_1 = data.get('artist_name_1')
    artist_name_2 = data.get('artist_name_2')

    if not artist_name_1 or not artist_name_2:
        return jsonify({'error': 'Both artist names are required'}), 400

    try:
        node1 = get_artist_node(SIMILARITY_GRAPH, artist_name_1)
        node2 = get_artist_node(SIMILARITY_GRAPH, artist_name_2)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if SIMILARITY_GRAPH.has_edge(node1, node2):
        weight = SIMILARITY_GRAPH[node1][node2]['weight']

    else:
        weight = -1.0

    return jsonify({'similarity_score': weight})


@app.route('/similarity-gat', methods=['POST'])
def similarity_gat():
    print("Get GAT similarity", flush=True)
    
    load_similarity_graph()
    load_gat_model()
    
    req_data = request.get_json()
    artist_name_1 = req_data.get('artist_name_1')
    artist_name_2 = req_data.get('artist_name_2')
    
    if not artist_name_1 or not artist_name_2:
        return jsonify({'error': 'Both artist names are required'}), 400

    if artist_name_1 not in node_id_to_idx or artist_name_2 not in node_id_to_idx:
        return jsonify({'error': 'Invalid artist name provided'}), 400

    idx_a = node_id_to_idx[artist_name_1]
    idx_b = node_id_to_idx[artist_name_2]

    edge_index = torch.tensor([[idx_a], [idx_b]], dtype=torch.long, device=gat_device)
    edge_attr = torch.ones(1, dtype=torch.float32, device=gat_device)

    inference_data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(node_id_to_idx)
    ).to(gat_device)

    with torch.no_grad():
        pred_score = gat_model(inference_data, return_attention=False)

    gat_score = pred_score.item()
    return jsonify({'gat_score': round(gat_score, 4)})

@app.route('/influence-diffusion', methods=['POST'])
def influence_diffusion():
    print("Get diffusion score", flush=True)
    load_influence_graph()
    data = request.get_json()
    artist_name_1 = data.get('artist_name_1')
    artist_name_2 = data.get('artist_name_2')

    if not artist_name_1 or not artist_name_2:
        return jsonify({'error': 'Both artist names are required'}), 400

    try:
        node1 = get_artist_node(INFLUENCE_GRAPH, artist_name_1)
        node2 = get_artist_node(INFLUENCE_GRAPH, artist_name_2)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if INFLUENCE_GRAPH.has_edge(node1, node2):
        weight = INFLUENCE_GRAPH[node1][node2]['weight']
    else:
        weight = -1.0

    return jsonify({'influence_score': weight})

@app.route('/influence-attention', methods=['POST'])
def influence_attention():
    load_attention_graph()
    print("Get attention score", flush=True)
    data = request.get_json()
    artist_name_1 = data.get('artist_name_1')
    artist_name_2 = data.get('artist_name_2')

    print("Artist names:", artist_name_1, artist_name_2, flush=True)

    if not artist_name_1 or not artist_name_2:
        print("Artist not existant", flush=True)
        return jsonify({'error': 'Both artist names are required'}), 400

    try:
        node1 = get_artist_node(ATTENTION_GRAPH, artist_name_1)
        node2 = get_artist_node(ATTENTION_GRAPH, artist_name_2)
    except ValueError as e:
        print(f"Error getting artist nodes: {e}", flush=True)
        return jsonify({'error': str(e)}), 400

    if ATTENTION_GRAPH.has_edge(node1, node2):
        print("Edge exists", flush=True)
        weight = ATTENTION_GRAPH[node1][node2]['weight']
    else:
        print("Edge non-existant", flush=True)
        weight = -1.0

    print("Returning weight", flush=True)
    print("Weight:", weight, flush=True)
    return jsonify({'attention_score': weight})

@app.route('/generate-graph', methods=['POST'])
def generate_graph():

    print("Call load graphs...", flush = True)
    load_graphs()

    data = request.get_json()
    artist_name_1 = data.get("artist_name_1")
    artist_name_2 = data.get("artist_name_2")
    graph_type = data.get("graph_type", "influence")

    if not artist_name_1 or not artist_name_2:
        return jsonify({"error": "Both artist names are required"}), 400

    if graph_type == "influence":
        return generate_influence_path_graph(artist_name_1, artist_name_2)
    elif graph_type == "explore1":
        return generate_exploration_graph(artist_name_1)
    elif graph_type == "explore2":
        return generate_exploration_graph(artist_name_2)
    else:
        return jsonify({"error": "Invalid graph type"}), 400

def generate_influence_path_graph(artist_name_1, artist_name_2):

    try:
        graph_filename = f"graph_{artist_name_1}_{artist_name_2}.html"
        graph_filepath = os.path.join(STATIC_DIR, graph_filename)
        
        if os.path.exists(graph_filepath):
            print("Graph already exists. Returning existing file.")
            graph_url = url_for('static', filename=graph_filename, _external=True)
            return jsonify({"graph_url": graph_url, "path_length": None})
        
        print("Generating new graph...")

        source_id = get_artist_node(SIMILARITY_GRAPH, artist_name_1)
        target_id = get_artist_node(SIMILARITY_GRAPH, artist_name_2)

        path = nx.shortest_path(DISTANCE_GRAPH, source=source_id, target=target_id, weight="weight")
        path_length = nx.shortest_path_length(DISTANCE_GRAPH, source=source_id, target=target_id, weight="weight")

        path_subgraph = SIMILARITY_GRAPH.subgraph(path)
        for i, node in enumerate(path):
            path_subgraph.nodes[node]["path_order"] = i

        print(f"Shortest path from {artist_name_1} to {artist_name_2}: {path}")
        print(f"Path length: {path_length}")
        print(f"Nodes in path subgraph: {list(path_subgraph.nodes())}")
        print(f"Edges in path subgraph: {list(path_subgraph.edges())}")

        net = Network(height="800px", width="100%", notebook=False, filter_menu=True, directed = True)
        net.from_nx(path_subgraph)

        for node in net.nodes:
            node['label'] = path_subgraph.nodes[node['id']].get('name', node['id'])
            node['font'] = {'size': FONT_SIZE}

        net.write_html(graph_filepath)
        __inline_resources(graph_filepath)
        print("Graph file saved at:", graph_filepath)

        graph_url = url_for('static', filename=graph_filename, _external=True)
        return jsonify({"graph_url": graph_url, "path_length": path_length})

    except nx.NetworkXNoPath:
        return jsonify({"error": f"No path exists between {artist_name_1} and {artist_name_2}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

def generate_exploration_graph(artist_name, max_distance=1.0):
    try:

        graph_filename = f"graph_explore_{artist_name}_{max_distance}.html"
        graph_filepath = os.path.join(STATIC_DIR, graph_filename)
        
        if os.path.exists(graph_filepath):
            print("Graph already exists. Returning existing file.")
            graph_url = url_for('static', filename=graph_filename, _external=True)
            return jsonify({"graph_url": graph_url})
        
        print("Generating new exploration graph...")

     
        source_id = get_artist_node(SIMILARITY_GRAPH, artist_name)

        total_nodes = len(DISTANCE_GRAPH.nodes())
        processed_nodes = 0
        nodes_within_distance = []
        start_time = time.time()

        print("Processing distances")
        distances = nx.single_source_dijkstra_path_length(DISTANCE_GRAPH, source=source_id, weight='weight')
        total_nodes = len(distances)  # More accurate count
        processed_nodes = 0

        for node, distance in distances.items():
            processed_nodes += 1
            if distance <= max_distance:
                nodes_within_distance.append(node)
                
            if processed_nodes % 20000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed_nodes}/{total_nodes} nodes "
                    f"({processed_nodes/total_nodes:.1%}), "
                    f"found {len(nodes_within_distance)} matches "
                    f"[{elapsed:.1f}s elapsed]", flush=True)

        if not nodes_within_distance:
            return jsonify({"error": f"No nodes found within distance {max_distance} from {artist_name}"}), 400

        print("Creating graph...")
        
        subgraph = SIMILARITY_GRAPH.subgraph(nodes_within_distance)
        if len(subgraph.nodes()) > MAX_NODES:
            nodes_within_distance = sorted(nodes_within_distance, key=lambda x: distances[x])[:MAX_NODES]
            subgraph = SIMILARITY_GRAPH.subgraph(nodes_within_distance)

        print(f"Subgraph size: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges", flush=True)

        net = Network(height="800px",
            width="100%",
            notebook=False,
            filter_menu=False,
            directed = True,
        )
        net.from_nx(subgraph)

        print("Net created")

        net.force_atlas_2based(
        )       


        for node in tqdm(net.nodes, desc="Formatting nodes", unit="node"):
            node_id = node['id']

            if node_id == source_id:
                node['color'] = {
                    'background': '#FF4136',
                    'border': '#85144b',
                    'highlight': {'background': '#FF0000', 'border': '#85144b'}
                }
            node['label'] = subgraph.nodes[node_id].get('name', node_id)
            
            degree = subgraph.degree[node_id]
            node['size'] = degree * 0.5 + 1
            node['font'] = {
                'size': FONT_SIZE,
                'strokeWidth': 3,

            }


        for edge in tqdm(net.edges, desc="Formatting edges", unit="edge"):  # Fixed description
            u = edge['from']
            v = edge['to']
            
            edge_data = subgraph.edges.get((u, v), {})
            similarity = edge_data.get('weight', 0.5)
            
            opacity = max(0.2, min(1.0, similarity))
            edge['color'] = f'rgba(100, 100, 100, {opacity})'
            edge['arrows'] = 'to'
            edge['arrowStrikethrough'] = False
            edge['smooth'] = False

        net.write_html(graph_filepath)
        __inline_resources(graph_filepath)
        print("Exploration graph saved at:", graph_filepath)

        graph_url = url_for('static', filename=graph_filename, _external=True)
        return jsonify({"graph_url": graph_url})

    except nx.NetworkXNoPath:
        return jsonify({"error": f"No paths exist from {artist_name}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# Not certain
def __inline_resources(html_path):
    """Reads the generated HTML and replaces external resource references with inlined content."""
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Define paths to your required library files from LIB_DIR.
    css_path = os.path.join(LIB_DIR, "tom-select", "tom-select.css")
    js_paths = [
        os.path.join(LIB_DIR, "tom-select", "tom-select.complete.min.js"),
        os.path.join(LIB_DIR, "bindings", "utils.js")
    ]

    # Replace the <link> tag for tom-select.css with an inline <style> tag.
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            css_content = css_file.read()
        html_content = html_content.replace(
            '<link href="lib/tom-select/tom-select.css" rel="stylesheet" />',
            f"<style>{css_content}</style>"
        )

    # Replace each <script> tag with the inlined JS.
    for js_rel_path in ["tom-select/tom-select.complete.min.js", "bindings/utils.js"]:
        js_file_path = os.path.join(LIB_DIR, *js_rel_path.split("/"))
        if os.path.exists(js_file_path):
            with open(js_file_path, "r", encoding="utf-8") as js_file:
                js_content = js_file.read()
            html_content = html_content.replace(
                f'<script src="lib/{js_rel_path}"></script>',
                f"<script>{js_content}</script>"
            )

    # Save the modified HTML back to disk.
    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_content)

if __name__ == '__main__':
    load_all_graphs()
    app.run(port=5000, debug=True)
    print("All graphs loaded", flush=True)