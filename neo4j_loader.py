import json
from neo4j import GraphDatabase

# --- CONFIG ---
NEO4J_URI = "neo4j+s://1d0cca9a.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "VVwrlfIgFASoThf5qb-vD-2r62HnNLuXthVzw8xnPPM"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# === Load JSON ===
def load_json_graph(file_path="data/generated_graph.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# === Flatten helper (to handle nested maps like progress_metric) ===
def flatten_props(props):
    flat = {}
    for k, v in props.items():
        if isinstance(v, dict):
            # flatten maps like progress_metric
            for subk, subv in v.items():
                flat[f"{k}_{subk}"] = subv
        elif isinstance(v, list):
            # flatten lists of simple values
            if all(isinstance(i, (str, int, float, bool)) for i in v):
                flat[k] = v
            elif all(isinstance(i, dict) for i in v):
                # for list of dicts that are not question_prompts
                flat[k] = [json.dumps(i, ensure_ascii=False) for i in v]
            else:
                flat[k] = str(v)
        else:
            flat[k] = v
    return flat


# === Recursive Creation ===
def create_nodes_and_relationships(tx, node, parent_id=None):
    node_id = node.get("id")
    node_label = "Concept"
    if node_id.startswith("P"):
        node_label = "Procedure"
    elif node_id.startswith("A"):
        node_label = "Assessment"

    # ✅ Flatten everything except children & connections
    props = flatten_props({k: v for k, v in node.items() if k not in ["children", "connections", "question_prompts"]})

    # Create node
    tx.run(
        f"""
        MERGE (n:{node_label} {{id:$id}})
        SET n += $props
        """,
        id=node_id,
        props=props
    )

    # === If Assessment has question_prompts, make Question nodes ===
    if node_label == "Assessment" and "question_prompts" in node:
        for idx, q in enumerate(node["question_prompts"], start=1):
            if isinstance(q, dict):
                q_text = q.get("question", "")
            else:
                q_text = str(q)
            q_id = f"{node_id}-Q{idx}"
            tx.run(
                """
                MERGE (q:Question {id:$qid})
                SET q.text = $text
                WITH q
                MATCH (a {id:$aid})
                MERGE (a)-[:HAS_QUESTION]->(q)
                """,
                qid=q_id,
                text=q_text,
                aid=node_id,
            )

    # === Create HAS_CHILD relationship if nested ===
    if parent_id:
        tx.run(
            """
            MATCH (p {id:$parent_id}), (c {id:$child_id})
            MERGE (p)-[:HAS_CHILD]->(c)
            """,
            parent_id=parent_id,
            child_id=node_id
        )

    # === Create connections between concepts ===
    for conn in node.get("connections", []):
        tx.run(
            f"""
            MATCH (a {{id:$from_id}}), (b {{id:$to_id}})
            MERGE (a)-[:{conn["relation"]}]->(b)
            """,
            from_id=node_id,
            to_id=conn["to"]
        )

    # === Recurse for children ===
    for child in node.get("children", []):
        create_nodes_and_relationships(tx, child, parent_id=node_id)


# === Main Load Function ===
def upload_to_neo4j(json_path="data/generated_graph.json"):
    graph = load_json_graph(json_path)
    with driver.session() as session:
        session.execute_write(create_nodes_and_relationships, graph)
    print("✅ Graph uploaded successfully to Neo4j!")


if __name__ == "__main__":
    upload_to_neo4j()
