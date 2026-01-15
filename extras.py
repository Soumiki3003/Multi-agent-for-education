"""
Supervisor Agent (Context Retrieval Layer)
------------------------------------------
GraphRAG (Neo4j) + Gemini embeddings + Gemini LLM.
Logs student interactions to student_state.json.
"""

import os
import json
import datetime
import neo4j
import re
import time
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.base import LLMInterface
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str

# === 1️⃣ Setup: Gemini API + Neo4j ===
GOOGLE_API_KEY = "AIzaSyDioekLy0lcRRMShudYIxvWUO_zY0_rZYc"
client = genai.Client(api_key=GOOGLE_API_KEY)

NEO4J_URI = "neo4j+s://1d0cca9a.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = "VVwrlfIgFASoThf5qb-vD-2r62HnNLuXthVzw8xnPPM"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# === 2️⃣ Embeddings using Gemini ===
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# === 3️⃣ Define a minimal Gemini-based LLM wrapper ===
# === 3️⃣ Define a minimal Gemini-based LLM wrapper ===
# class GeminiLLM(LLMInterface):
#     """Gemini-based LLM wrapper compatible with Neo4j GraphRAG."""

#     def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
#         self.client = genai.Client(api_key=api_key)
#         self.model = model

#     # Required synchronous call
#     def invoke(self, prompt: str) -> str:
#         response = self.client.models.generate_content(
#             model=self.model,
#             contents=prompt
#         )
#         return response.text if hasattr(response, "text") else str(response)

#     # Required async call (simple sync wrapper)
#     async def ainvoke(self, prompt: str) -> str:
#         return self.invoke(prompt)

#     # Required __call__ method for compatibility
#     def __call__(self, prompt: str) -> str:
#         return self.invoke(prompt)


# class GeminiLLM:
#     def __init__(self, api_key: str):
#         self.client = genai.Client(api_key=api_key)
#         self.model_name = "gemini-2.0-flash"

#     def invoke(self, prompt: str, message_history=None, system_instruction=None):
#         """
#         Executes a Gemini LLM query. Compatible with GraphRAG.
#         """
#         try:
#             # Combine system + user prompt if provided
#             if system_instruction:
#                 full_prompt = f"{system_instruction}\n\nUser Query:\n{prompt}"
#             else:
#                 full_prompt = prompt

#             response = self.client.models.generate_content(
#                 model=self.model_name,
#                 contents=full_prompt
#             )

#             # Safely extract text
#             text = ""
#             if hasattr(response, "text"):
#                 text = response.text
#             elif hasattr(response, "candidates"):
#                 text = response.candidates[0].content.parts[0].text
#             else:
#                 text = str(response)

#             # ✅ Return as GraphRAG-compatible object
#             return LLMResponse(content=text)

#         except Exception as e:
#             return LLMResponse(content=f"[GeminiLLM Error] {e}")

class GeminiLLM:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        """Initialize Gemini client for GraphRAG-compatible LLM calls."""
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def invoke(self, prompt: str, message_history=None, system_instruction=None, max_retries: int = 6):
        """
        Generates content using Gemini with retry & backoff logic for RESOURCE_EXHAUSTED errors.
        Returns LLMResponse(content=str)
        """
        full_prompt = f"{system_instruction}\n\nUser Query:\n{prompt}" if system_instruction else prompt

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )

                # Extract text from Gemini’s response safely
                if hasattr(response, "text"):
                    text = response.text
                elif hasattr(response, "candidates"):
                    text = response.candidates[0].content.parts[0].text
                else:
                    text = str(response)

                return LLMResponse(content=text)

            except Exception as e:
                err_msg = str(e)

                # Handle rate-limit / quota errors gracefully
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    # Try to parse "retry in XXs"
                    match = re.search(r"retry in (\d+)", err_msg)
                    wait_time = int(match.group(1)) if match else 2 ** attempt
                    print(f"⚠️  Gemini quota hit. Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue  # retry again
                else:
                    print(f"❌  Gemini error (non-quota): {e}")
                    return LLMResponse(content=f"[GeminiLLM Error] {e}")

        # If all retries fail
        return LLMResponse(content="[GeminiLLM Error] Max retries reached (quota still exhausted)")



# === 4️⃣ Initialize VectorRetriever + GraphRAG ===
retriever = VectorRetriever(
    driver=driver,
    index_name="vectorIndexCTF",
    embedder=embeddings
)

llm = GeminiLLM("AIzaSyDioekLy0lcRRMShudYIxvWUO_zY0_rZYc")
graph_rag = GraphRAG(retriever, llm)



# === 5️⃣ Student State Management ===
STATE_PATH = os.path.join("data", "student_state.json")
os.makedirs("data", exist_ok=True)

def load_student_state():
    if not os.path.exists(STATE_PATH):
        return {"students": []}
    with open(STATE_PATH, "r") as f:
        return json.load(f)

def save_student_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# === 6️⃣ Context Retrieval ===
def retrieve_context(student_id: str, query: str):
    print(f"\n🔍 Retrieving context for: '{query}' (student={student_id})")

    result = graph_rag.search(
        query_text=query,
        retriever_config={"top_k": 5},
        return_context=True
    )

    retrieved_context = getattr(result, "context", [])
    state = load_student_state()

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "student_id": student_id,
        "query": query,
        "retrieved_nodes": [c["text"] for c in retrieved_context] if retrieved_context else [],
        "scores": [c.get("score") for c in retrieved_context] if retrieved_context else [],
        "interaction_type": "context_retrieval"
    }

    # Append or create student trajectory
    for s in state["students"]:
        if s["id"] == student_id:
            s["trajectory"].append(entry)
            break
    else:
        state["students"].append({"id": student_id, "trajectory": [entry]})

    save_student_state(state)
    print("🧭 Context retrieval logged successfully.")
    return result


# === 7️⃣ Test run ===
if __name__ == "__main__":
    print("\n🚀 Testing Supervisor Agent Context Retrieval (GraphRAG + Gemini)")
    sample_query = "Explain how to initialize angr project."
    result = retrieve_context("student_001", sample_query)

    # === Extract only the concise LLM answer ===
    clean_answer = (
        result.answer.strip()
        if hasattr(result, "answer") and isinstance(result.answer, str)
        else str(getattr(result, "answer", result))
    )

    print("\n Supervisor Agent Response (student view, will be delivered via Companion Agent):\n")
    print(clean_answer)

    # Optional: log full retriever output for debugging / instructor analysis
    with open("logs/context_debug.txt", "a") as f:
        f.write(f"\n\n[Query: {sample_query}]\n")
        f.write(f"Answer: {clean_answer}\n")
        f.write(f"Retriever Context: {getattr(result, 'retriever_result', 'N/A')}\n")
