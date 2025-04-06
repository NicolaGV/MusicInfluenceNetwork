# app.py
from flask import Flask, request, jsonify, url_for, g
from flask_cors import CORS
import uuid
import os
import networkx as nx
from pyvis.network import Network
import time
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "similarity_graph.gexf")
INFLUENCE_GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "influence_graph.gexf")
ATTENTION_GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "attention_composite.gexf")


FONT_SIZE = 25
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
LIB_DIR = os.path.join(APP_DIR, "lib")
MAX_NODES = 200


app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Precomputed graphs
G = None
H = None
DISTANCE_GRAPH_PATH = os.path.join(os.path.dirname(GRAPH_PATH), "distance_graph.gexf")

INFLUENCE_GRAPH = None
ATTENTION_GRAPH = None

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def load_graphs():
    if 'graphs_loaded' not in g:
        g.graphs_loaded = True
        g.G = nx.read_gexf(GRAPH_PATH)

        # Load or generate distance graph
        if os.path.exists(DISTANCE_GRAPH_PATH):
            print("Loading precomputed distance graph...")
            g.H = nx.read_gexf(DISTANCE_GRAPH_PATH)
        else:
            print("Generating and saving distance graph...", flush=True)
            g.H = nx.DiGraph()
            g.H.add_nodes_from(g.G.nodes(data=True))

            total_edges = len(g.G.edges())
            processed_edges = 0
            print(f"Total edges to process: {total_edges}")

            for u, v, data in g.G.edges(data=True):
                if "weight" in data:
                    g.H.add_edge(u, v, weight=1.0 - data["weight"])
                processed_edges += 1

                if processed_edges % 100000 == 0:
                    print(f"Processed {processed_edges}/{total_edges} edges", flush=True)
            nx.write_gexf(g.H, DISTANCE_GRAPH_PATH)

def load_all_graphs():
    load_influence_graph()
    load_attention_graph()

def load_influence_graph():
    if 'influence_loaded' not in g:
        print("Loading influence graph...", flush=True)
        g.influence_loaded = True
        g.INFLUENCE_GRAPH = nx.read_gexf(INFLUENCE_GRAPH_PATH)

def load_attention_graph():
    if 'attention_loaded' not in g:
        print("Loading attention graph...", flush=True)
        g.attention_loaded = True
        g.ATTENTION_GRAPH = nx.read_gexf(ATTENTION_GRAPH_PATH)


def get_artist_node(G: nx.DiGraph, artist_name: str):
    nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("name") == artist_name]
    if not nodes:
        raise ValueError(f"Artist '{artist_name}' not found in graph")
    if len(nodes) > 1:
        raise ValueError(f"Multiple nodes found for '{artist_name}'")
    return nodes[0][0]

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
        node1 = get_artist_node(g.INFLUENCE_GRAPH, artist_name_1)
        node2 = get_artist_node(g.INFLUENCE_GRAPH, artist_name_2)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if g.INFLUENCE_GRAPH.has_edge(node1, node2):
        weight = g.INFLUENCE_GRAPH[node1][node2]['weight']
    # elif g.G.has_edge(node2, node1): # Want directed influence, so commented
    #     weight = g.G[node2][node1]['weight']
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
        node1 = get_artist_node(g.ATTENTION_GRAPH, artist_name_1)
        node2 = get_artist_node(g.ATTENTION_GRAPH, artist_name_2)
    except ValueError as e:
        print(f"Error getting artist nodes: {e}", flush=True)
        return jsonify({'error': str(e)}), 400

    if g.ATTENTION_GRAPH.has_edge(node1, node2):
        print("Edge exists", flush=True)
        weight = g.ATTENTION_GRAPH[node1][node2]['weight']
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

        source_id = get_artist_node(g.G, artist_name_1)
        target_id = get_artist_node(g.G, artist_name_2)

        path = nx.shortest_path(g.H, source=source_id, target=target_id, weight="weight")
        path_length = nx.shortest_path_length(g.H, source=source_id, target=target_id, weight="weight")

        path_subgraph = g.G.subgraph(path)
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

     
        source_id = get_artist_node(g.G, artist_name)

        total_nodes = len(g.H.nodes())
        processed_nodes = 0
        nodes_within_distance = []
        start_time = time.time()

        print("Processing distances")
        distances = nx.single_source_dijkstra_path_length(g.H, source=source_id, weight='weight')
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
        
        subgraph = g.G.subgraph(nodes_within_distance)
        if len(subgraph.nodes()) > MAX_NODES:
            nodes_within_distance = sorted(nodes_within_distance, key=lambda x: distances[x])[:MAX_NODES]
            subgraph = g.G.subgraph(nodes_within_distance)

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
    app.run(port=5000, debug=True)
    load_all_graphs()
