# app.py
from flask import Flask, request, jsonify, send_from_directory, send_file, url_for
from flask_cors import CORS
import uuid
import os
import networkx as nx
from pyvis.network import Network

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "..", "..", "similarity_graph.gexf")
FONT_SIZE = 25
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
LIB_DIR = os.path.join(APP_DIR, "lib")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def make_distance_graph(G: nx.DiGraph):
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if "weight" in data:
            H.add_edge(u, v, weight=1.0 - data["weight"])
    return H

def get_artist_node(G: nx.DiGraph, artist_name: str):
    nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("name") == artist_name]
    if not nodes:
        raise ValueError(f"Artist '{artist_name}' not found in graph")
    if len(nodes) > 1:
        raise ValueError(f"Multiple nodes found for '{artist_name}'")
    return nodes[0][0]

@app.route('/influence-diffusion', methods=['POST'])
def influence_diffusion():
    data = request.get_json()
    artist_name_1 = data.get('artist_name_1')
    artist_name_2 = data.get('artist_name_2')

    if not artist_name_1 or not artist_name_2:
        return jsonify({'error': 'Both artist names are required'}), 400

    influence_score = 0.16

    return jsonify({'influence_score': influence_score})

@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    data = request.get_json()
    artist_name_1 = data.get("artist_name_1")
    artist_name_2 = data.get("artist_name_2")

    if not artist_name_1 or not artist_name_2:
        return jsonify({"error": "Both artist names are required"}), 400

    try:
        graph_filename = f"graph_{artist_name_1}_{artist_name_2}.html"
        graph_filepath = os.path.join(STATIC_DIR, graph_filename)
        
        if os.path.exists(graph_filepath):
            print("Graph already exists. Returning existing file.")
            graph_url = url_for('static', filename=graph_filename, _external=True)
            return jsonify({"graph_url": graph_url, "path_length": None})
        
        print("Generating new graph...")
        G = nx.read_gexf(GRAPH_PATH)
        distance_graph = make_distance_graph(G)

        source_id = get_artist_node(G, artist_name_1)
        target_id = get_artist_node(G, artist_name_2)

        path = nx.shortest_path(distance_graph, source=source_id, target=target_id, weight="weight")
        path_length = nx.shortest_path_length(distance_graph, source=source_id, target=target_id, weight="weight")

        path_subgraph = G.subgraph(path)
        for i, node in enumerate(path):
            path_subgraph.nodes[node]["path_order"] = i

        print(f"Shortest path from {artist_name_1} to {artist_name_2}: {path}")
        print(f"Path length: {path_length}")
        print(f"Nodes in path subgraph: {list(path_subgraph.nodes())}")
        print(f"Edges in path subgraph: {list(path_subgraph.edges())}")

        net = Network(height="800px", width="100%", notebook=False, filter_menu=True)
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
