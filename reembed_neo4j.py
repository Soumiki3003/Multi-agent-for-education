# === reembed_neo4j.py ===
"""
Re-embeds all nodes in Neo4j using Gemini embeddings (3072-dim)
and stores the new vectors into the `embedding` property.
"""

import json
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
from langchain_google_genai._common import GoogleGenerativeAIError

# === Neo4j connection ===
NEO4J_URI = "neo4j+s://1d0cca9a.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "VVwrlfIgFASoThf5qb-vD-2r62HnNLuXthVzw8xnPPM"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# === Gemini embeddings ===
API_KEY = "AIzaSyDchhG7QSTBD0qnHWmVzcUh5sIOAMUslBo"
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

def reembed_nodes():
    with driver.session() as session:
        query = """
        MATCH (n)
        WHERE n:Concept OR n:Procedure OR n:Assessment
        RETURN n.id AS id, n.name AS name, n.definition AS definition
        """
        records = list(session.run(query))
        print(f"🔍 Found {len(records)} nodes to re-embed")

        for i, record in enumerate(records, 1):
            node_id = record["id"]
            text = f"{record['name']} {record.get('definition','')}"
            while True:
                try:
                    vector = embeddings.embed_query(text)
                    break
                except GoogleGenerativeAIError as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        print("⚠️  Quota hit, sleeping 65 s…")
                        time.sleep(65)
                        continue
                    raise e

            session.run(
                "MATCH (n {id:$id}) SET n.embedding=$embedding",
                id=node_id, embedding=vector)
            print(f"✅  [{i}/{len(records)}] {node_id} re-embedded")
            time.sleep(0.7)  # gentle throttle

if __name__ == "__main__":
    reembed_nodes()
