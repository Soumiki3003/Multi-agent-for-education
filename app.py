# === app.py ===
from flask import Flask, render_template, request, jsonify
import json, os
from parsers.dual_parser import parse_dualpath
from neo4j import GraphDatabase
import re

# === Flask setup ===
app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
DATA_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# === Neo4j setup ===
NEO4J_URI = "neo4j+s://1d0cca9a.databases.neo4j.io"  # or neo4j+s://<your-db>.databases.neo4j.io for Aura
NEO4J_USER = "neo4j"
NEO4J_PASS = "VVwrlfIgFASoThf5qb-vD-2r62HnNLuXthVzw8xnPPM"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# === ROUTE: Login Page ===
@app.route("/")
def login():
    return render_template("login.html")


# === ROUTE: File Upload + Dual Parser ===
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        html_link = request.form.get("html_link", "")
        uploaded_paths = []

        for f in uploaded_files:
            if f and f.filename:
                path = os.path.join(UPLOAD_DIR, f.filename)
                f.save(path)
                uploaded_paths.append(path)
                print(f"📄 Uploaded: {path}")

                # Run dual-path parser
                generated = parse_dualpath(path)
                print(f"Generated KG JSON: {generated}")

        return jsonify({
            "uploaded": uploaded_paths,
            "html_link": html_link,
            "graph_generated": True
        })

    return render_template("upload.html")


# === ROUTE: Graph Visualization Page ===
@app.route("/graph")
def graph():
    return render_template("graph.html")


# === ROUTE: Graph Data (Now From Neo4j) ===
@app.route("/graph-data")
def graph_data():
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    nodes, links = {}, []

    with driver.session() as session:
        results = session.run(query)
        for record in results:
            n = record["n"]
            r = record["r"]
            m = record["m"]

            # Add source and target nodes
            if n["id"] not in nodes:
                nodes[n["id"]] = {
                    "id": n["id"],
                    "name": n.get("name", ""),
                    "label": list(n.labels)[0],
                    **{k: v for k, v in dict(n).items() if k not in ["id", "name"]}
                }
            if m["id"] not in nodes:
                nodes[m["id"]] = {
                    "id": m["id"],
                    "name": m.get("name", ""),
                    "label": list(m.labels)[0],
                    **{k: v for k, v in dict(m).items() if k not in ["id", "name"]}
                }

            # Add edge
            links.append({
                "source": n["id"],
                "target": m["id"],
                "type": r.type if hasattr(r, "type") else "HAS_CHILD"
            })

    graph_json = {"nodes": list(nodes.values()), "links": links}
    return jsonify(graph_json)

# === ROUTE: Root-level Conceptual Nodes (for initial render) ===
@app.route("/graph-root")
def graph_root():
    """
    Returns a root node ("Central node") with top-level Concept nodes as children.
    This mimics the previous hierarchical structure for expandable D3 behavior.
    """
    with driver.session() as session:
        # Get top-level Concept nodes (no incoming HAS_CHILD)
        result = session.run("""
            MATCH (c:Concept)
            WHERE NOT (()-[:HAS_CHILD]->(c))
            RETURN c
        """)
        concept_nodes = [dict(r["c"]) for r in result]

    # Build a pseudo-root node just like your previous JSON
    root = {
        "id": "root",
        "name": "Central node",
        "children": concept_nodes
    }
    return jsonify(root)


# === ROUTE: Expand Node on Demand ===
@app.route("/expand-node/<node_id>")
def expand_node(node_id):
    """
    Expands a single node (Concept/Procedure/Assessment) by fetching its immediate children.
    This lets the D3 graph dynamically add new layers on click.
    """
    with driver.session() as session:
        results = session.run("""
            MATCH (n {id:$node_id})-[:HAS_CHILD|PROCEDURAL_FOR|ASSESSES]->(child)
            RETURN child
        """, node_id=node_id)
        children = [dict(r["child"]) for r in results]
    return jsonify(children)


# === ROUTE: Update Node (optional for later editing) ===
# @app.route("/update_node", methods=["POST"])
# def update_node():
#     node = request.get_json()
#     path = os.path.join(DATA_DIR, "generated_graph.json")
#     if not os.path.exists(path):
#         return jsonify({"error": "Graph not found"}), 404

#     with open(path) as f:
#         graph = json.load(f)

#     for i, n in enumerate(graph.get("children", [])):
#         if n["id"] == node["id"]:
#             graph["children"][i] = node
#             break

#     with open(path, "w") as f:
#         json.dump(graph, f, indent=2)

#     return jsonify({"status": "updated"})

# @app.route("/update_node", methods=["POST"])
# def update_node():
#     node = request.get_json()
#     with driver.session() as session:
#         query = """
#         MATCH (n {id: $id})
#         SET n += $props
#         RETURN n
#         """
#         props = {k: v for k, v in node.items() if k not in ["id", "children"]}
#         session.run(query, id=node["id"], props=props)
#     return jsonify({"status": "updated"})

@app.route("/update_node", methods=["POST"])
def update_node():
    node = request.get_json()
    props = {k: v for k, v in node.items() if k not in ["id", "children"]}

    # 🔧 Automatically fix "source" to link to the real uploaded file
    if "source" in props and isinstance(props["source"], str):
        match = re.search(r'([A-Za-z0-9_\-]+)\s*\[page', props["source"])
        if match:
            base_name = match.group(1).strip()
            props["source"] = f"/static/uploads/{base_name}.pdf"

    with driver.session() as session:
        query = """
        MATCH (n {id: $id})
        SET n += $props
        RETURN n
        """
        session.run(query, id=node["id"], props=props)

    return jsonify({"status": "updated"})



# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
